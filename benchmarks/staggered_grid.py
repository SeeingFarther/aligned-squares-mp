import random
import heapq
import gc
from math import sqrt, ceil
import networkx as nx

from discopygal.geometry_utils import collision_detection, conversions
from discopygal.bindings import *
from discopygal.solvers.Solver import Solver
from discopygal.solvers import Scene, PathCollection, Path, PathPoint
from discopygal.solvers.nearest_neighbors import NearestNeighbors
from discopygal.solvers.metrics import Metric, Metric_Euclidean


class NeighborsFinder:
    def __init__(self):
        self.tree = Ss.Kd_tree()

    def fit(self, points):
        self.tree.insert(points)

    def neighbors_in_radius(self, query, r):
        epsilon = Ker.FT(0)
        a = Ss.Fuzzy_sphere(query, r.to_double(), epsilon.to_double())
        res = self.tree.search(a)
        return res


class PrmNode:
    def __init__(self, pt, name):
        self.point = Ss.Point_d(2, [pt.x().to_double(), pt.y().to_double()])
        self.point_2 = pt
        self.name = name
        self.out_connections = {}
        self.in_connections = {}


class PrmEdge:
    def __init__(self, src, dest, eid):
        self.src = src
        self.dest = dest
        self.name = "e" + str(eid)
        self.segment = Ker.Segment_2(src.point_2, dest.point_2)
        self.cost = sqrt(Ker.squared_distance(src.point_2, dest.point_2).to_double())


class PrmGraph:
    def __init__(self, milestones, writer):
        self.edge_id = 0
        self.points_to_nodes = {}
        self.edges = []
        for milestone in milestones:
            if milestone in self.points_to_nodes:
                print("Error, re entry of:", milestone.point, "in", milestone, file=writer)
            self.points_to_nodes[milestone.point] = milestone

    def insert_edge(self, milestone, neighbor):
        neighbor_node = self.points_to_nodes[neighbor]
        if neighbor_node == milestone:
            return
        if neighbor_node in milestone.out_connections or neighbor_node in milestone.in_connections:
            return
        self.edges.append(PrmEdge(milestone, neighbor_node, self.edge_id))
        milestone.out_connections[neighbor_node] = self.edges[-1]
        neighbor_node.in_connections[milestone] = self.edges[-1]
        self.edge_id += 1

    def a_star(self, robot_radius, start_points, goal_points, writer):
        goal = tuple([self.points_to_nodes[p] for p in goal_points])

        def h(p):
            res = 0
            for p1, p2 in zip(p, goal):
                res += sqrt(Ker.squared_distance(p1.point_2, p2.point_2).to_double())
            return res

        def get_neighbors(points):
            connections_list = [list(p.out_connections.keys()) + list(p.in_connections.keys()) for p in points]
            res = []
            for i, _ in enumerate(points):
                is_good = True
                for next_point in connections_list[i]:
                    for j, _ in enumerate(points):
                        if i == j:
                            continue
                        if Ker.squared_distance(next_point.point_2, points[j].point_2) < Ker.FT(
                                4) * robot_radius * robot_radius:
                            is_good = False
                            break
                    if not is_good:
                        continue
                    seg = points[i].in_connections[next_point] if next_point in points[i].in_connections else \
                        points[i].out_connections[next_point]
                    for j, _ in enumerate(points):
                        if i == j:
                            continue
                        if Ker.squared_distance(seg.segment, points[j].point_2) < Ker.FT(
                                4) * robot_radius * robot_radius:
                            is_good = False
                            break
                    if not is_good:
                        continue
                    res.append((tuple([points[k] if k != i else next_point for k in range(len(points))]), seg))
            return res

        temp_i = 0
        start = tuple([self.points_to_nodes[p] for p in start_points])

        def get_path(cf):
            c = goal
            path = [[p.point_2 for p in c]]
            while c != start:
                c = cf[c]
                path.append([p.point_2 for p in c])
            path.reverse()
            return path

        q = [(h(start), temp_i, start)]
        heapq.heapify(q)
        came_from = {}
        g_score = {start: 0}
        # temp_j = 0
        while len(q) > 0:
            curr_f_score, _, curr = heapq.heappop(q)
            if curr_f_score > (g_score[curr] + h(curr)):
                # temp_j += 1
                # if temp_j % 100000 == 0:
                #     print("temp_j", temp_j, file=writer)
                #     print(len(q), " ", len(came_from), file=writer)
                continue
            if curr == goal:
                return g_score[curr], get_path(came_from)
            for neighbor, edge in get_neighbors(curr):
                tentative_g_score = g_score[curr] + edge.cost
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = curr
                    g_score[neighbor] = tentative_g_score
                    temp_i += 1
                    if temp_i % 100000 == 0:
                        print("Added", temp_i, "items to A_star heap", file=writer)
                        # print(len(q), " ", len(came_from), file=writer)
                        if temp_i % 1000000 == 0:
                            # TODO remove duplications?
                            gc.collect()
                    heapq.heappush(q, (tentative_g_score + h(neighbor), temp_i, neighbor))
        print("error no path found", file=writer)
        return None, []


def cords_to_points_2(cords_x, cords_y):
    res = [Ker.Point_2(Ker.FT(x), Ker.FT(y)) for x in cords_x for y in cords_y]
    return res


class StaggeredGrid(Solver):
    def __init__(self, eps, delta, bounding_margin_width_factor=Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
                 nearest_neighbors=None, metric=None, sampler=None):
        super().__init__(bounding_margin_width_factor)
        self.nearest_neighbors: NearestNeighbors = nearest_neighbors
        if self.nearest_neighbors is None:
            self.nearest_neighbors = NeighborsFinder()

        self.metric: Metric = metric
        if self.metric is None:
            self.metric = Metric_Euclidean

        self.roadmap = None
        self.collision_detection = None
        self.eps = eps
        self.delta = delta  # Clearance after the environment is shrunken to [0,1]^2
        self.is_multi_robot = True
        self.sample_method = "staggered_grid"
        self.reset()

    def reset(self):
        alpha = self.eps / sqrt(1 + self.eps ** 2)
        y = self.eps / (2 * (2 + self.eps))
        self.edge_len = 1 - 2 * self.delta
        if self.is_multi_robot:
            # These may change as we modify bounds in the paper
            self.ball_radius = y * self.delta
            # self.connection_radius = Ker.FT(self.delta)*(Ker.FT(1+self.eps)/Ker.FT(2+self.eps))
            self.connection_radius = Ker.FT(self.delta) * (Ker.FT(1 + self.eps) / Ker.FT(2 + self.eps)) * Ker.FT(1.0001)
            # self.connection_radius = Ker.FT(self.delta*1.001)
        else:
            self.ball_radius = alpha * self.delta
            self.connection_radius = Ker.FT(2 * (alpha + sqrt(1 - alpha ** 2)) * self.delta) * Ker.FT(1.0001)
        unrounded_balls_per_dim = self.edge_len / (2 * self.ball_radius)
        print("unrounded number of balls:", unrounded_balls_per_dim ** 2 + (unrounded_balls_per_dim + 1) ** 2)
        self.balls_per_dim = ceil(unrounded_balls_per_dim)
        self.num_of_sample_points = self.balls_per_dim ** 2 + (self.balls_per_dim + 1) ** 2
        print("real number of balls:", self.num_of_sample_points)
        self.grid_points_per_dim = ceil(sqrt(self.num_of_sample_points))
        print("real number of grid points:", self.grid_points_per_dim ** 2)

    def generate_milestones(self):
        def conv(num):
            return (self.max - self.min) * num + self.min

        res = []
        if self.sample_method == "staggered_grid":
            i = 0
            half_points_diff = (self.edge_len / self.balls_per_dim) / 2
            l1_cords = [conv(self.delta + (2 * i - 1) * half_points_diff) for i in range(1, self.balls_per_dim + 1)]
            l2_cords = [conv(self.delta + (2 * i) * half_points_diff) for i in range(self.balls_per_dim + 1)]
            l1 = cords_to_points_2(l1_cords, l1_cords)
            l2 = cords_to_points_2(l2_cords, l2_cords)
            all_points = l1 + l2
            for point in all_points:
                if self.collision_detection.is_point_valid(point) and \
                        self._bounding_box.min_x <= point.x() <= self._bounding_box.max_x and \
                        self._bounding_box.min_y <= point.y() <= self._bounding_box.max_y:
                    res.append(PrmNode(point, "v" + str(i)))
                    i += 1
            return res

        if self.sample_method == "grid":
            i = 0
            points_diff = (self.edge_len / (self.grid_points_per_dim - 1))
            cords = [conv(self.delta + i * points_diff) for i in range(self.grid_points_per_dim)]
            points = cords_to_points_2(cords, cords)
            for point in points:
                if self.collision_detection.is_point_valid(point):
                    res.append(PrmNode(point, "v" + str(i)))
                    i += 1
            return res

        if self.sample_method == "random":
            i = 0
            points = [Ss.Point_d(2, [Ker.FT(random.uniform(self.min + self.delta, self.max - self.delta)),
                                     Ker.FT(random.uniform(self.min + self.delta, self.max - self.delta))]) for _ in
                      range(self.num_of_sample_points)]
            for point in points:
                if self.collision_detection.is_point_valid(point):
                    res.append(PrmNode(point, "v" + str(i)))
                    i += 1
            return res

        raise ValueError("Invalid configuration")

    def make_graph(self, milestones):
        g = PrmGraph(milestones, self.writer)
        for milestone in milestones:
            p = milestone.point
            nearest = self.nearest_neighbors.neighbors_in_radius(p,
                                                                 self.connection_radius * Ker.FT(self.max - self.min))
            for neighbor in nearest:
                if neighbor == p:
                    continue
                edge = Ker.Segment_2(Ker.Point_2(p[0], p[1]), Ker.Point_2(neighbor[0], neighbor[1]))
                if self.collision_detection.is_edge_valid(edge):
                    g.insert_edge(milestone, neighbor)

        return g

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
            'eps': ('Epsilon:', 9999, int),
            'delta': ('Delta:', 0.04, float),
            'bounding_margin_width_factor': (
                'Margin width factor (for bounding box):', Solver.DEFAULT_BOUNDS_MARGIN_FACTOR, FT),
        }

    def get_graph(self):
        """
        Return a graph (if applicable).
        Can be overridded by solvers.

        :return: graph whose vertices are Point_2 or Point_d
        :rtype: :class:`networkx.Graph` or :class:`None`
        """
        roadmap = nx.Graph()
        roadmap.add_nodes_from(list(self.g.points_to_nodes.keys()))
        for edge in self.g.edges:
            roadmap.add_edge(edge.src.point, edge.dest.point)
        return roadmap

    def collision_free(self, p, q):
        """
        Get two points in the configuration space and decide if they can be connected
        """
        p_list = conversions.Point_d_to_Point_2_list(p)
        q_list = conversions.Point_d_to_Point_2_list(q)

        # Check validity of each edge seperately
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

    def load_scene(self, scene: Scene):
        """
        Load a scene into the solver.
        Also build the roadmap.

        :param scene: scene to load
        :type scene: :class:`~discopygal.solvers.Scene`
        """
        super().load_scene(scene)
        # self.sampler.set_scene(scene, self._bounding_box)

        self.reset()

        self.sources = [robot.start for robot in scene.robots]
        self.destinations = [robot.end for robot in scene.robots]

        # Get squares robots edges length
        radii = set()
        for robot in scene.robots:
            for e in robot.poly.edges():
                radii.add( Metric_Euclidean.dist(e.source(), e.target()).to_double())
                break

        if len(radii) == 0:
            self.log("Error: must have at least one robot")
        elif len(radii) > 1:
            self.log("Error: this only works when all robots are of the same radii")
        self.robot_length = radii[0]

        self.collision_detection = collision_detection.ObjectCollisionDetection(scene.obstacles, scene.robots[0])

        if self._bounding_box.min_x != self._bounding_box.min_y or self._bounding_box.max_x != self._bounding_box.max_y:
            self.log("Error: Must be used on square scenes")

        self.max = max(self._bounding_box.max_x, self._bounding_box.max_y)
        self.min = min(self._bounding_box.min_x, self._bounding_box.min_y)

        milestones = []
        for (i, start_p) in enumerate(self.sources):
            milestones.append(PrmNode(start_p, "start" + str(i)))
        for (i, destination_p) in enumerate(self.destinations):
            milestones.append(PrmNode(destination_p, "goal" + str(i)))
        milestones += self.generate_milestones()

        self.nearest_neighbors.fit([milestone.point for milestone in milestones])
        self.g = self.make_graph(milestones)
        self.log(f"Vertices amount per robot:{len(self.g.points_to_nodes)}")
        self.log(f"Edges amount per robot: {len(self.g.edges)}")

    def solve(self):
        """
        Based on the start and end locations of each robot, solve the scene
        (i.e. return paths for all the robots)

        :return: path collection of motion planning
        :rtype: :class:`~discopygal.solvers.PathCollection`
        """
        radius = Ker.FT(self.robot_length)
        path_collection = PathCollection()
        a_star_res, d_path = self.g.a_star(
            radius,
            [Ss.Point_d(2, [start_p.x().to_double(), start_p.y().to_double()]) for start_p in self.sources],
            [Ss.Point_d(2, [destination_p.x().to_double(), destination_p.y().to_double()]) for destination_p in
             self.destinations],
            self.writer)

        self.log(f"A_star_res: {a_star_res}")
        if a_star_res is None:
            return path_collection

        self.log('successfully found a path...')
        for robot in self.scene.robots:
            path_collection.add_robot_path(robot, Path([]))

        for point in d_path:
            for i, robot in enumerate(self.scene.robots):
                path_collection.paths[robot].points.append(PathPoint(point[i]))

        # return path, G
        return path_collection


class BasicsStaggeredGridForExperiments(StaggeredGrid):
    def load_scene(self, scene: Scene):
        super().load_scene(scene)

    def solve(self):
        path = super().solve()
        return path
