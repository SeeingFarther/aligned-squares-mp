from discopygal.bindings import *
from discopygal.geometry_utils import collision_detection
from discopygal.solvers.metrics import Metric_Euclidean


class GapPositionFinder:

    def __init__(self, scene):

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

    def find_gap_positions_y(self, x, y_intersections, robot_index):
        free_positions = []
        y_intersections.sort()
        length = len(y_intersections)
        robot_length = self.robot_lengths[robot_index]
        for i in range(length - 1):
            diff = y_intersections[i + 1] - y_intersections[i]
            offset = robot_length * 0.5
            add = y_intersections[i] + (abs(y_intersections[i + 1]) - abs(y_intersections[i])) * 0.5
            position = Point_2(FT(x - offset), FT(add - offset))
            if diff >= robot_length and self.collision_detection[self.robots[robot_index]].is_point_valid(position):
                free_positions.append(position)

        return free_positions

    def find_gap_positions_x(self, y, x_intersections, robot_index):
        free_positions = []
        x_intersections.sort()
        length = len(x_intersections)
        robot_length = self.robot_lengths[robot_index]
        for i in range(length - 1):
            diff = x_intersections[i + 1] - x_intersections[i]
            offset = robot_length * 0.5
            add = x_intersections[i] + (abs(x_intersections[i + 1]) - abs(x_intersections[i])) * 0.5
            position = Point_2(FT(add - offset), FT(y - offset))
            if diff >= robot_length and self.collision_detection[self.robots[robot_index]].is_point_valid(position):
                free_positions.append(position)

        return free_positions
