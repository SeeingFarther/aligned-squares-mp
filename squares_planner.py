import sys
import json
import networkx as nx

from discopygal.solvers.samplers import Sampler
from discopygal.solvers import Scene
from discopygal.solvers import PathPoint, Path, PathCollection
from discopygal.solvers.metrics import Metric_Euclidean
from discopygal.solvers.nearest_neighbors import NearestNeighbors_sklearn
from discopygal.bindings import *
from discopygal.geometry_utils import collision_detection, conversions
from discopygal.solvers.Solver import Solver

from samplers.space_sampler import SpaceSampler
from utils.utils import get_point_d, find_max_value_coordinates, get_robot_point_by_idx
from metrics.ctd_metric import Metric_CTD
from metrics.epsilon_metric import Metric_Epsilon_2, Metric_Epsilon_Inf
from utils.nearest_neighbors import NearestNeighbors_sklearn_ball
from samplers.pair_sampler import CombinedSquaresSampler


class SquareMotionPlanner(Solver):
    def __init__(self, num_landmarks: int, k: int,
                 bounding_margin_width_factor: FT = Solver.DEFAULT_BOUNDS_MARGIN_FACTOR, sampler: Sampler = None):
        super().__init__(FT(bounding_margin_width_factor))
        self.num_landmarks = num_landmarks

        if k > num_landmarks:
            print('Error: Number of K bigger than number of landmarks')
            exit(-1)

        self.k = k
        self.sampler = sampler
        if self.sampler is None:
            self.sampler = SpaceSampler()

        metric = 'CTD'
        if metric is None or metric == 'Euclidean':
            self.nearest_neighbors = NearestNeighbors_sklearn()
        elif metric == 'CTD':
            self.nearest_neighbors = NearestNeighbors_sklearn_ball(Metric_CTD)
        elif metric == 'Epsilon_2':
            self.nearest_neighbors = NearestNeighbors_sklearn_ball(Metric_Epsilon_2)
        elif metric == 'Epsilon_Inf':
            self.nearest_neighbors = NearestNeighbors_sklearn_ball(Metric_Epsilon_Inf)
        else:
            print('Unknown metric')
            exit(-1)

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

    def collision_free(self, p: Point_d, q: Point_d) -> bool:
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

    def _single_robot_collision_free(self, robot, point, neighbor):
        edge = Segment_2(point, neighbor)
        return self.collision_detection[robot].is_edge_valid(edge)

    def add_edge_func(self, point: Point_2, neighbor: Point_2, graph: nx.Graph):
        """
        Add an edge between two points in the roadmap
        """

        point_1_coords = Point_2(point[0], point[1])
        point_2_coords = Point_2(point[2], point[3])
        neighbor_1_coords = Point_2(neighbor[0], neighbor[1])
        neighbor_2_coords = Point_2(neighbor[2], neighbor[3])

        # Compute distance
        dist_1 = self.metric.dist(point_1_coords, neighbor_1_coords).to_double()
        dist_2 = self.metric.dist(point_2_coords, neighbor_2_coords).to_double()

        # Add edge to graph
        graph.add_edge(point, neighbor, weight=dist_1 + dist_2)
        return

    def merge_path_points(self, orig_tensor_path: list):
        new_tensor_path: list[Point_d] = [orig_tensor_path[0]]
        for idx, curr_joint_point in enumerate(orig_tensor_path[1: len(orig_tensor_path) - 1], 1):
            robots_shorten_size = [[-1, -1], [-1, -1]]
            prev_joint_point = new_tensor_path[idx - 1]
            orig_curr_joint_point = orig_tensor_path[idx]
            next_joint_point = orig_tensor_path[idx + 1]
            for robot_idx in [0, 1]:

                # Robot indices
                other_robot_idx = 1 - robot_idx
                robot = self.scene.robots[robot_idx]
                other_robot = self.scene.robots[other_robot_idx]

                # Point we try to remove from robot path
                orig_curr_robot_point = get_robot_point_by_idx(orig_curr_joint_point, robot_idx)

                # Neighboring points to curr point in the path
                prev_robot_point = get_robot_point_by_idx(prev_joint_point, robot_idx)
                next_robot_point = get_robot_point_by_idx(next_joint_point, robot_idx)

                # Compute previous edge size
                prev_edge_size = self.metric.dist(prev_robot_point, orig_curr_robot_point).to_double()
                next_edge_size = self.metric.dist(orig_curr_robot_point, next_robot_point).to_double()
                prev_path_size = prev_edge_size + next_edge_size

                for prev_next_idx, new_curr_joint_point in enumerate([prev_joint_point, next_joint_point]):
                    new_curr_robot_point = get_robot_point_by_idx(new_curr_joint_point, robot_idx)

                    # Check if we can connect the previous robot point to the new current point for the robot.
                    if not self._single_robot_collision_free(robot, prev_robot_point, new_curr_robot_point):
                        continue
                    # Check if we can connect the new current robot point to the next point for the robot.
                    if not self._single_robot_collision_free(robot, new_curr_robot_point, next_robot_point):
                        continue

                    # Check if previous edges of the robots don't collide
                    prev_robot_edge = Segment_2(prev_robot_point, new_curr_robot_point)
                    prev_other_robot_point = get_robot_point_by_idx(prev_joint_point, other_robot_idx)
                    curr_other_robot_point = get_robot_point_by_idx(orig_curr_joint_point, other_robot_idx)
                    prev_other_robot_edge = Segment_2(prev_other_robot_point, curr_other_robot_point)
                    if collision_detection.collide_two_robots(robot, prev_robot_edge, other_robot,
                                                              prev_other_robot_edge):
                        continue

                    # Check if next edges of the robots don't collide
                    next_robot_edge = Segment_2(new_curr_robot_point, next_robot_point)
                    next_other_robot_point = get_robot_point_by_idx(next_joint_point, other_robot_idx)
                    next_other_robot_edge = Segment_2(curr_other_robot_point, next_other_robot_point)
                    if collision_detection.collide_two_robots(robot, next_robot_edge, other_robot,
                                                              next_other_robot_edge):
                        continue

                    new_prev_edge_size = self.metric.dist(prev_robot_point, new_curr_robot_point).to_double()
                    new_next_edge_size = self.metric.dist(new_curr_robot_point, next_robot_point).to_double()
                    new_path_size = new_prev_edge_size + new_next_edge_size

                    robots_shorten_size[robot_idx][prev_next_idx] = prev_path_size - new_path_size

            # Select which robot point to remove from path from the joint point of the robots by selecting the one
            # which shorten our path the most
            robot_idx_to_shorten, prev_next_idx_to_shorten = find_max_value_coordinates(robots_shorten_size)
            if robots_shorten_size[robot_idx_to_shorten][prev_next_idx_to_shorten] <= 0:
                solution_joint_point = orig_curr_joint_point
            else:
                solution_joint_point = get_point_d(robot_idx_to_shorten, prev_next_idx_to_shorten, prev_joint_point,
                                                   orig_curr_joint_point, next_joint_point)
            new_tensor_path.append(solution_joint_point)

        new_tensor_path.append(orig_tensor_path[-1])
        return new_tensor_path

    def remove_path_points(self, orig_tensor_path: list):

        # Pass the path and try to get rid of unnecessary joint robots points to shorten path
        new_tensor_path: list[Point_d] = [orig_tensor_path[0]]
        robot_0 = self.scene.robots[0]
        robot_1 = self.scene.robots[1]
        for idx, curr_joint_point in enumerate(orig_tensor_path[1: len(orig_tensor_path) - 1], 1):
            prev_joint_point = new_tensor_path[-1]
            next_joint_point = orig_tensor_path[idx + 1]

            # Can skip the current point in robot 0 without collision?
            prev_robot_0_point = get_robot_point_by_idx(prev_joint_point, 0)
            next_robot_0_point = get_robot_point_by_idx(next_joint_point, 0)
            robot_0_collision_free = self._single_robot_collision_free(robot_0, prev_robot_0_point, next_robot_0_point)

            # Can skip the current point in robot 1 without collision?
            prev_robot_1_point = get_robot_point_by_idx(prev_joint_point, 1)
            next_robot_1_point = get_robot_point_by_idx(next_joint_point, 1)
            robot_1_collision_free = self._single_robot_collision_free(robot_1, prev_robot_1_point, next_robot_1_point)

            # Edges don't collide?
            robot_0_edge = Segment_2(prev_robot_0_point, next_robot_0_point)
            robot_1_edge = Segment_2(prev_robot_1_point, next_robot_1_point)
            robots_collision_free = not collision_detection.collide_two_robots(robot_0, robot_0_edge, robot_1,
                                                                               robot_1_edge)

            # All the requirements fulfilled? Skip both robot points
            if robot_0_collision_free and robot_1_collision_free and robots_collision_free:
                continue

            new_tensor_path.append(curr_joint_point)

        new_tensor_path.append(orig_tensor_path[-1])
        return new_tensor_path

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
        return SquareMotionPlanner(d['num_landmarks'], d['k'], FT(d['bounding_margin_width_factor']))

    def load_scene(self, scene: Scene):
        """
        Load a scene into the solver.
        Also build the roadmap.

        :param scene: scene to load
        :type scene: :class:`~discopygal.solvers.Scene`
        """
        super().load_scene(scene)
        self.sampler.set_num_samples(self.num_landmarks)
        self.sampler.set_scene(scene)

        # Build collision detection for each robot
        for i, robot in enumerate(scene.robots):
            self.collision_detection[robot] = collision_detection.ObjectCollisionDetection(scene.obstacles, robot)

        ################
        # Build the Planner
        ################
        self.roadmap = nx.Graph()

        self.sampler.set_scene(scene, self._bounding_box)
        self.sampler.set_num_samples(self.num_landmarks)
        self.sampler.ready_sampler()

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
        G = nx.Graph()
        G.add_nodes_from(self.roadmap.nodes(data=True))
        for cnt, point in enumerate(self.roadmap.nodes):
            neighbors = self.nearest_neighbors.k_nearest(point, self.k + 1)
            for neighbor in neighbors:
                if neighbor == point:
                    continue

                middle_point_coords = conversions.Point_2_list_to_Point_d(
                    [Point_2(point[0], point[1]), Point_2(neighbor[2], neighbor[3])])
                if self.collision_free(neighbor, point):
                    # Add edge to graph
                    self.add_edge_func(point, neighbor, G)

                # Try 3 Phase movement instead
                elif self.collision_free(neighbor, middle_point_coords) and self.collision_free(middle_point_coords,
                                                                                                point):
                    G.add_node(middle_point_coords)
                    self.add_edge_func(middle_point_coords, neighbor, G)
                    self.add_edge_func(point, middle_point_coords, G)

            if cnt % 100 == 0 and self.verbose:
                print('connected', cnt, 'landmarks to their nearest neighbors', file=self.writer)

        self.roadmap = nx.Graph(G)

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
        tensor_path = self.merge_path_points(self.remove_path_points(tensor_path))
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
    with open('scene_length_1.json', 'r') as fp:
        scene = Scene.from_dict(json.load(fp))
    solver = SquareMotionPlanner(num_landmarks=1000, k=2,sampler=SpaceSampler())
    solver.load_scene(scene)
    solver.solve()
