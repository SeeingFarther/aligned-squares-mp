import networkx as nx
import json
from samplers.samplers import CombinedSquaresSampler
from discopygal.solvers import Scene
from discopygal.solvers import PathPoint, Path, PathCollection
from discopygal.solvers.metrics import Metric_Euclidean
from discopygal.solvers.nearest_neighbors import NearestNeighbors_sklearn
from discopygal.bindings import *
from discopygal.geometry_utils import collision_detection, conversions
from discopygal.solvers.Solver import Solver


class SquareMotionPlanner(Solver):
    def __init__(self, num_landmarks, k, combined, bounding_margin_width_factor=Solver.DEFAULT_BOUNDS_MARGIN_FACTOR):
        super().__init__(FT(bounding_margin_width_factor))
        self.num_landmarks = num_landmarks

        if k > num_landmarks:
            print('Error: Number of K bigger than number of landmarks')
            exit(-1)

        self.k = k
        self.combined = combined
        self.nearest_neighbors = NearestNeighbors_sklearn()
        self.sampler = None
        self.metric = Metric_Euclidean
        self.roadmap = None
        self.collision_detection = {}
        self.start = None
        self.end = None

    def sample_free(self):
        """
        Sample a free random sample for both robot 1 and robot 2 independently
        """

        # Sampling Separate
        p_rand = []
        for robot in self.scene.robots:
            sample = self.sampler.sample()
            while not self.collision_detection[robot].is_point_valid(sample):
                sample = self.sampler.sample()
            p_rand.append(sample)
        p_rand = conversions.Point_2_list_to_Point_d(p_rand)
        return p_rand

    def collision_free(self, p, q):
        """
        Get two points in the configuration space and decide if they can be connected
        """
        p_list = conversions.Point_d_to_Point_2_list(p)
        q_list = conversions.Point_d_to_Point_2_list(q)

        # Check validity of each edge separately
        for i, robot in enumerate(self.scene.robots):
            edge = Segment_2(p_list[i], q_list[i])
            if not self.collision_detection[robot].is_edge_valid(edge):
                return False

        # Check validity of coordinated robot motion
        for i, robot1 in enumerate(self.scene.robots):
            for j, robot2 in enumerate(self.scene.robots):
                if j <= i:
                    continue
                edge1 = Segment_2(p_list[i], q_list[i])
                edge2 = Segment_2(p_list[j], q_list[j])
                if collision_detection.collide_two_robots(robot1, edge1, robot2, edge2):
                    return False

        return True

    def get_graph(self):
        """
        Return a graph (if applicable).
        Can be overridden by solvers.

        :return: graph whose vertices are Point_2 or Point_d
        :rtype: :class:`networkx.Graph` or :class:`None`
        """
        return self.roadmap

    @staticmethod
    def get_arguments():
        """
        Return a list of arguments and their description, defaults and types.
        Can be used by a GUI to generate fields dynamically.
        Should be overridded by solvers.

        :return: arguments dict
        :rtype: :class:`dict`
        """
        return {
            'num_landmarks': ('Number of Landmarks:', 1000, int),
            'k': ('K for nearest neighbors:', 15, int),
            'combined': ('Combined sampling or Separate sampling:', 1, int),
            'bounding_margin_width_factor': ('Margin width factor (for bounding box):', 0, FT),
        }

    @staticmethod
    def from_arguments(d):
        """
        Get a dictionary of arguments and return a solver.
        Should be overridded by solvers.

        :param d: arguments dict
        :type d: :class:`dict`
        """
        return SquareMotionPlanner(d['num_landmarks'], d['k'], d['combined'], FT(d['bounding_margin_width_factor']))

    def load_scene(self, scene: Scene):
        """
        Load a scene into the solver.
        Also build the roadmap.

        :param scene: scene to load
        :type scene: :class:`~discopygal.solvers.Scene`
        """
        super().load_scene(scene)

        # Build collision detection for each robot
        for i, robot in enumerate(scene.robots):
            self.collision_detection[robot] = collision_detection.ObjectCollisionDetection(scene.obstacles, robot)

        self.sampler = CombinedSquaresSampler(scene, self._bounding_box)

        ################
        # Build the Planner
        ################
        self.roadmap = nx.Graph()

        # Add start & end points
        self.start = conversions.Point_2_list_to_Point_d([robot.start for robot in scene.robots])
        self.end = conversions.Point_2_list_to_Point_d([robot.end for robot in scene.robots])
        self.roadmap.add_node(self.start)
        self.roadmap.add_node(self.end)

        # Add valid points
        for i in range(self.num_landmarks):
            p_rand = self.sampler.sample_free()
            self.roadmap.add_node(p_rand)
            if i % 100 == 0 and self.verbose:
                print('added', i, 'landmarks in PRM', file=self.writer)

        self.nearest_neighbors.fit(list(self.roadmap.nodes))

        # Connect all points to their k nearest neighbors
        for cnt, point in enumerate(self.roadmap.nodes):
            neighbors = self.nearest_neighbors.k_nearest(point, self.k + 1)
            for neighbor in neighbors:
                if neighbor == point:
                    continue
                if self.collision_free(neighbor, point):
                    # Coordinates
                    point_1_coords = Point_2(point[0], point[1])
                    point_2_coords = Point_2(point[2], point[3])
                    neighbor_1_coords = Point_2(neighbor[0], neighbor[1])
                    neighbor_2_coords = Point_2(neighbor[2], neighbor[3])

                    # Compute distance
                    dist_1 = self.metric.dist(point_1_coords, neighbor_1_coords).to_double()
                    dist_2 = self.metric.dist(point_2_coords, neighbor_2_coords).to_double()

                    # Add edge to graph
                    self.roadmap.add_edge(point, neighbor, weight=dist_1 + dist_2)

            if cnt % 100 == 0 and self.verbose:
                print('connected', cnt, 'landmarks to their nearest neighbors', file=self.writer)

    def solve(self):
        """
        Based on the start and end locations of each robot, solve the scene
        (i.e. return paths for all the robots)

        :return: path collection of motion planning
        :rtype: :class:`~discopygal.solvers.PathCollection`
        """
        if not nx.algorithms.has_path(self.roadmap, self.start, self.end):
            if self.verbose:
                print('no path found...', file=self.writer)
            return PathCollection()

        # Convert from a sequence of Point_d points to PathCollection
        tensor_path = nx.algorithms.shortest_path(self.roadmap, self.start, self.end, weight='weight')
        path_collection = PathCollection()
        for i, robot in enumerate(self.scene.robots):
            points = []
            for point in tensor_path:
                points.append(PathPoint(Point_2(point[2 * i], point[2 * i + 1])))
            path = Path(points)
            path_collection.add_robot_path(robot, path)

        if self.verbose:
            print('successfully found a path...', file=self.writer)

        return path_collection


if __name__ == '__main__':
    with open('test.json', 'r') as fp:
        scene = Scene.from_dict(json.load(fp))
    solver = SquareMotionPlanner(num_landmarks=100, k=1, combined=1)
    solver.load_scene(scene)
    solver.solve()
