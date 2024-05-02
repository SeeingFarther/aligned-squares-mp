from discopygal.bindings import *
from discopygal.geometry_utils import collision_detection
from discopygal.solvers import Scene
from discopygal.solvers.metrics import Metric_Euclidean

from .utils import euclidean_distance_1d


class GapPositionFinder:
    """
    Find gap positions in a scene
    """

    def __init__(self, scene: Scene):
        """
        Constructor

        :param scene:
        :type scene: :class:`~discopygal.solvers.Scene`
        """

        self.robot_lengths = [0, 0]
        self.robots = []
        self.obstacles = scene.obstacles
        self.collision_detection = {}

        # Build collision detection for each robot
        for i, robot in enumerate(scene.robots):
            self.robots.append(robot)
            self.collision_detection[robot] = collision_detection.ObjectCollisionDetection(scene.obstacles, robot)

            # Get squares robots edges length
            for e in robot.poly.edges():
                self.robot_lengths[i] = Metric_Euclidean.dist(e.source(), e.target()).to_double()
                break

    def find_gap_positions_y(self, x: float, y_intersections: list, robot_index: int) -> list:
        """
        Find the gap positions in the y-axis incase obstacles are present inside
        the square at the length of sum of the robot lengths
        :param x:
        :type x: float
        :param y_intersections:
        :type y_intersections: list
        :param robot_index:
        :type robot_index: int

        :return: list of free positions
        :rtype: list
        """
        free_positions = []
        y_intersections.sort()
        length = len(y_intersections)
        robot_length = self.robot_lengths[robot_index]
        for i in range(length - 1):
            # Length between two intersections
            distance_between_points = euclidean_distance_1d(y_intersections[i], y_intersections[i + 1])

            # Distance is smaller than the robot length? Continue
            if distance_between_points < robot_length:
                continue

            # Find middle point between two intersections
            middle_point = y_intersections[i] + distance_between_points * 0.5

            # Offset points as our reference is the left bottom point of the robot
            offset = robot_length * 0.5
            x_position = x - offset
            y_position = middle_point - offset

            # Check if position is valid
            position = Point_2(FT(x_position), FT(y_position))
            if self.collision_detection[self.robots[robot_index]].is_point_valid(position):
                free_positions.append(position)

        return free_positions

    def find_gap_positions_x(self, y: float, x_intersections: list, robot_index: int) -> list:
        """
        Find the gap positions in the x-axis incase obstacles are present inside the square at the length of sum of
        the robot lengths
        :param y:
        :type y: float
        :param x_intersections:
        :type x_intersections: list
        :param robot_index:
        :type robot_index: int

        :return: list of free positions
        :rtype: list
        """

        free_positions = []
        x_intersections.sort()
        length = len(x_intersections)
        robot_length = self.robot_lengths[robot_index]
        for i in range(length - 1):

            # Length between two intersections
            distance_between_points = euclidean_distance_1d(x_intersections[i], x_intersections[i + 1])

            # Distance is smaller than the robot length? Continue
            if distance_between_points < robot_length:
                continue

            # Find middle point between two intersections
            middle_point = x_intersections[i] + distance_between_points * 0.5

            # Offset points as our reference is the left bottom point of the robot
            offset = robot_length * 0.5
            x_position = middle_point - offset
            y_position = y - offset

            # Check if position is valid
            position = Point_2(FT(x_position), FT(y_position))
            if self.collision_detection[self.robots[robot_index]].is_point_valid(position):
                free_positions.append(position)

        return free_positions
