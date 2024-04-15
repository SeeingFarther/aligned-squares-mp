import networkx
from discopygal.geometry_utils import collision_detection
from discopygal.solvers import Scene
from discopygal.solvers.metrics import Metric_Euclidean, Metric
from discopygal.solvers.samplers import Sampler_Uniform

from utils.utils import *
from utils.gap_position_finder import GapPositionFinder


class BasicSquaresSampler(Sampler_Uniform):
    def __init__(self, scene: Scene, metric: Metric = None, graph: networkx.Graph = None):
        # Init
        super().__init__(scene)
        self.gap_finder = None
        self.square_length = None
        self.robot_lengths = [0, 0]
        self.robots = []
        self.metric = metric
        self.roadmap = graph

        if scene is None:
            return

        self.set_scene(scene)

    def find_trivial_positions(self, square_center, robot_index: int) -> list:
        """
        Find the free positions the robot can be placed inside the square
        :return list of free trivial positions for robot inside the square:
        :rtype list:
        """

        # Find corners
        center_x = square_center.x().to_double()
        center_y = square_center.y().to_double()
        corners = find_square_corners(self.square_length, center_x, center_y)

        # Find positions inside the square that are collision free
        free_positions = []
        diff = self.robot_lengths[robot_index] * 0.5
        for corner in corners:
            position_x = corner[0] + diff if corner[0] < center_x else corner[0] - diff
            position_y = corner[1] + diff if corner[1] < center_y else corner[1] - diff
            position_x -= diff
            position_y -= diff
            position = Point_2(FT(position_x), FT(position_y))
            if self.collision_detection[self.robots[robot_index]].is_point_valid(position):
                free_positions.append(position)

        return free_positions

    def find_non_trivial_y_positions(self, square_center: Point_2, robot_index: int) -> list:
        """
        Find the free positions the robot can be placed inside the square
        :return list of free trivial positions for robot inside the square:
        :rtype list:
        """

        # Find corners
        center_x = square_center.x().to_double()
        center_y = square_center.y().to_double()

        # Find limits
        diff = self.square_length * 0.5
        min_y = center_y - diff
        max_y = center_y + diff
        min_x = center_x - diff
        max_x = center_x + diff

        # Find positions inside the square that are collision free
        diff = self.robot_lengths[robot_index] * 0.5
        free_positions = []
        x = min_x + diff
        y_intersections = compute_y_intersections(x, min_y, max_y, self.obstacles)
        y_intersections += [min_y, max_y]
        y_intersections = list(set(y_intersections))
        free_positions += self.gap_finder.find_gap_positions_y(x, y_intersections, robot_index)

        x = max_x - diff
        y_intersections = compute_y_intersections(x, min_y, max_y, self.obstacles)
        y_intersections += [min_y, max_y]
        y_intersections = list(set(y_intersections))
        free_positions += self.gap_finder.find_gap_positions_y(x, y_intersections, robot_index)

        return free_positions

    def find_non_trivial_x_positions(self, square_center: Point_2, robot_index: int):
        # Find corners
        center_x = square_center.x().to_double()
        center_y = square_center.y().to_double()

        # Find limits
        diff = self.square_length * 0.5
        min_y = center_y - diff
        max_y = center_y + diff
        min_x = center_x - diff
        max_x = center_x + diff

        # Find positions inside the square that are collision free
        diff = self.robot_lengths[robot_index] * 0.5
        free_positions = []
        y = min_y + diff
        x_intersections = compute_x_intersections(y, min_x, max_x, self.obstacles)
        x_intersections += [min_x, max_x]
        x_intersections = list(set(x_intersections))
        free_positions += self.gap_finder.find_gap_positions_x(y, x_intersections, robot_index)

        # Find positions inside the square that are collision free
        y = max_y - diff
        x_intersections = compute_x_intersections(y, min_x, max_x, self.obstacles)
        x_intersections += [min_x, max_x]
        x_intersections = list(set(x_intersections))
        free_positions += self.gap_finder.find_gap_positions_x(y, x_intersections, robot_index)

        return free_positions

    def set_scene(self, scene, bounding_box = None):
        """
        Set the scene the sampler should use.
        Can be overridded to add additional processing.

        :param scene: a scene to sample in
        :type scene: :class:`~discopygal.solvers.Scene`
        """
        super().set_scene(scene, bounding_box)
        self.obstacles = scene.obstacles
        self.collision_detection = {}

        # Build collision detection for each robot
        i = 0
        for robot in scene.robots:
            self.robots.append(robot)
            self.collision_detection[robot] = collision_detection.ObjectCollisionDetection(scene.obstacles, robot)

            # Get squares robots edges length
            for e in robot.poly.edges():
                self.robot_lengths[i] = Metric_Euclidean.dist(e.source(), e.target()).to_double()
                i += 1
                break

        # Length of the square we try to fit
        self.square_length = sum(self.robot_lengths)
        self.gap_finder = GapPositionFinder(scene)
