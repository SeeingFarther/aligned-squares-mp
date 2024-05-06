import json
import networkx as nx
import numpy as np
from discopygal.geometry_utils.bounding_boxes import calc_scene_bounding_box
from discopygal.geometry_utils.conversions import Point_2_list_to_Point_d, Point_2_to_xy, FT_to_float

from discopygal.solvers.samplers import Sampler
from discopygal.solvers import Scene, Robot
from discopygal.solvers import PathPoint, Path, PathCollection
from discopygal.solvers.metrics import Metric_Euclidean
from discopygal.solvers.nearest_neighbors import NearestNeighbors_sklearn
from discopygal.bindings import *
from discopygal.geometry_utils import collision_detection, conversions
from discopygal.solvers.Solver import Solver

from samplers.basic_sampler import BasicSquaresSampler
from samplers.bridge_sampler import BridgeSampler
from samplers.combined_sampler import CombinedSampler
from samplers.middle_sampler import MiddleSampler
from samplers.pair_sampler import PairSampler
from samplers.space_sampler import SpaceSampler
from utils.path_shortener import PathShortener
from metrics.ctd_metric import Metric_CTD
from metrics.epsilon_metric import Metric_Epsilon_2, Metric_Epsilon_Inf
from utils.nearest_neighbors import NearestNeighbors_sklearn_ball
from utils.utils import point2_to_point_d


class SquaresPrm(Solver):
    """
    Squares PRM Solver
    """

    def __init__(self, num_landmarks: int, k: int,
                 bounding_margin_width_factor: FT = Solver.DEFAULT_BOUNDS_MARGIN_FACTOR, sampler: Sampler = None):
        """
        Constructor for the SquaresPrm solver.
        :param num_landmarks:
        :type num_landmarks: int
        :param k:
        :type k: int
        :param bounding_margin_width_factor:
        :type bounding_margin_width_factor: FT
        :param samplers:
        :type samplers: list[:class:`~discopygal.solvers.samplers.Sampler`]
        """
        super().__init__(FT(bounding_margin_width_factor))
        self.num_landmarks = num_landmarks

        if k > num_landmarks:
            print('Error: Number of K bigger than number of landmarks')
            exit(-1)

        # Set samplers
        samplers = [SpaceSampler(), BridgeSampler(), MiddleSampler(y_axis=True), MiddleSampler(y_axis=False)]
        #samplers = [MiddleSampler(y_axis=True), MiddleSampler(y_axis=True)]
        num_prob = [1 / len(samplers)] * len(samplers)
        self.combined_sampler = CombinedSampler(num_prob, samplers)

        # Choose metric for nearest neighbors
        metric = 'Euclidean'
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

        # Metric for distance computation
        self.metric = Metric_Euclidean
        self.k = k
        self.roadmap = None
        self.collision_detection = {}
        self.start = None
        self.end = None

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
        return SquaresPrm(d['num_landmarks'], d['k'], FT(d['bounding_margin_width_factor']))

    def collision_free(self, p: Point_d, q: Point_d) -> bool:
        """
        Get two points in the configuration space and decide if they can be connected
        without colliding with any obstacles.
        :param p: start point
        :type p: :class:`~discopygal.bindings.Point_d`
        :param q: end point
        :type q: :class:`~discopygal.bindings.Point_d`

        :return: True if the edge is collision free, False otherwise
        :rtype: bool
        """

        # Convert to Point_2 list
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

    def is_edge_valid(self, robot: Robot, p: Point_2, q: Point_2) -> bool:
        """
        Check if the edge between the point and the neighbor is collision free for the robot
        :param robot:
        :type robot: :class:`Robot`
        :param p:
        :type p: :class:`~discopygal.bindings.Point_2`
        :param q:
        :type q: :class:`~discopygal.bindings.Point_2`

        :return: True if the edge is collision free, False otherwise
        :rtype: bool
        """
        is_edge_valid = True
        edge = Segment_2(p, q)
        if not self.collision_detection[robot].is_edge_valid(edge):
            is_edge_valid = False
        return is_edge_valid

    def add_edge_func(self, point: Point_2, neighbor: Point_2, graph: nx.Graph):
        """
        Add an edge between two points in the roadmap
        :param point:
        :type point: :class:`~discopygal.bindings.Point_2`
        :param neighbor:
        :type neighbor: :class:`~discopygal.bindings.Point_2`
        :param graph:
        :type graph: :class:`networkx.Graph`

        """

        # Build the points
        point_1_coords = Point_2(point[0], point[1])
        point_2_coords = Point_2(point[2], point[3])
        neighbor_1_coords = Point_2(neighbor[0], neighbor[1])
        neighbor_2_coords = Point_2(neighbor[2], neighbor[3])

        # Compute distance of the edge
        dist_1 = self.metric.dist(point_1_coords, neighbor_1_coords).to_double()
        dist_2 = self.metric.dist(point_2_coords, neighbor_2_coords).to_double()

        # Add edge to graph
        graph.add_edge(point, neighbor, weight=dist_1 + dist_2)
        return

    def load_scene(self, scene: Scene):
        """
        Load a scene into the solver.
        Also build the roadmap.

        :param scene: scene to load
        :type scene: :class:`~discopygal.solvers.Scene`
        """
        super().load_scene(scene)
        self.combined_sampler.set_scene(scene, self._bounding_box)
        self.combined_sampler.set_num_samples(self.num_landmarks)
        self.combined_sampler.ready_sampler()

        # Build collision detection for each robot
        for i, robot in enumerate(scene.robots):
            self.collision_detection[robot] = collision_detection.ObjectCollisionDetection(scene.obstacles, robot)

        ################
        # Build the Planner
        ################
        self.roadmap = nx.Graph()

        # Add start & end points
        self.start = conversions.Point_2_list_to_Point_d([robot.start for robot in scene.robots])
        self.end = conversions.Point_2_list_to_Point_d([robot.end for robot in scene.robots])
        self.roadmap.add_node(self.start)
        self.roadmap.add_node(self.end)

        self.roadmaps = [nx.Graph()] * len(self.collision_detection)
        self.starts = []
        self.ends = []
        for i, robot in enumerate(scene.robots):
            self.starts.append(robot.start)
            self.ends.append(robot.end)
            self.roadmaps[i].add_node(robot.start)
            self.roadmaps[i].add_node(robot.end)

        # Add valid points
        nearest_neighbors = NearestNeighbors_sklearn()
        min_x, max_x, min_y, max_y = self._bounding_box or calc_scene_bounding_box(self.scene)
        min_x, max_x, min_y, max_y = FT_to_float(min_x), FT_to_float(max_x), FT_to_float(min_y), FT_to_float(max_y)
        for j in range(self.num_landmarks):
            p_rand = []
            for i, robot in enumerate(scene.robots):
                p_rand.append(self.combined_sampler.sample_free(i))
                nearest_neighbors.fit(list(self.roadmaps[i].nodes))
                point = p_rand[i]
                neighbor = nearest_neighbors.k_nearest(point, 1)[0]
                n_x, n_y = Point_2_to_xy(neighbor)
                p_x, p_y = Point_2_to_xy(point)
                reward = (np.sqrt(((p_x - n_x) ** 2))) / (max_x - min_x)
                reward += (np.sqrt(((p_y - n_y) ** 2))) / (max_y - min_y)
                self.combined_sampler.update_probs(i, reward)
                self.roadmaps[i].add_node(point)

        sampler = PairSampler()
        sampler.set_scene(scene, self._bounding_box)
        landmarks = int(0.0 * self.num_landmarks)
        sampler.set_num_samples(landmarks)
        sampler.ready_sampler()

        for j in range(self.num_landmarks - landmarks):
            p_rand = []
            for i, robot in enumerate(scene.robots):
                p_rand.append(self.combined_sampler.sample_free(i))
            p_rand = conversions.Point_2_list_to_Point_d(p_rand)

            self.roadmap.add_node(p_rand)
            if j % 100 == 0 and self.verbose:
                print('added', j, 'landmarks in PRM', file=self.writer)

        for j in range(landmarks):
            p_rand = (sampler.sample_free())
            self.roadmap.add_node(p_rand)

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
                second_middle_point_coords = conversions.Point_2_list_to_Point_d(
                    [Point_2(neighbor[0], neighbor[1]), Point_2(point[2], point[3])])
                if self.collision_free(neighbor, point):
                    # Add edge to graph
                    self.add_edge_func(point, neighbor, G)

                # # Try 3 Phase movement instead
                elif self.collision_free(neighbor, middle_point_coords) and self.collision_free(middle_point_coords,
                                                                                                point):
                    G.add_node(middle_point_coords)
                    self.add_edge_func(middle_point_coords, neighbor, G)
                    self.add_edge_func(point, middle_point_coords, G)

                elif self.collision_free(neighbor, second_middle_point_coords) and self.collision_free(
                        second_middle_point_coords,
                        point):
                    G.add_node(second_middle_point_coords)
                    self.add_edge_func(second_middle_point_coords, neighbor, G)
                    self.add_edge_func(point, second_middle_point_coords, G)

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
        tensor_path = PathShortener(self.scene, self.metric).shorten_path(tensor_path)
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
    with open('./scenes/cubic3.json', 'r') as fp:
        scene = Scene.from_dict(json.load(fp))
    solver = SquaresPrm(num_landmarks=1000, k=15, sampler=None)
    solver.load_scene(scene)
    solver.solve()
