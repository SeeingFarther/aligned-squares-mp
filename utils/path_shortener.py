from discopygal.bindings import Segment_2, Point_d
from discopygal.geometry_utils import collision_detection
from discopygal.solvers import Scene
from discopygal.solvers.metrics import Metric

from utils.utils import get_robot_point_by_idx, get_point_d, find_max_value_coordinates


class PathShortener:
    def __init__(self, scene: Scene, metric: Metric):
        self.robots = scene.robots
        self.scene = scene
        self.metric = metric

        self.collision_detection = {}
        for i, robot in enumerate(scene.robots):
            self.collision_detection[robot] = collision_detection.ObjectCollisionDetection(scene.obstacles, robot)

    def single_robot_collision_free(self, robot, point, neighbor):
        edge = Segment_2(point, neighbor)
        return self.collision_detection[robot].is_edge_valid(edge)

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
                robot = self.robots[robot_idx]
                other_robot = self.robots[other_robot_idx]

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
                    if not self.single_robot_collision_free(robot, prev_robot_point, new_curr_robot_point):
                        continue
                    # Check if we can connect the new current robot point to the next point for the robot.
                    if not self.single_robot_collision_free(robot, new_curr_robot_point, next_robot_point):
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
        robot_0 = self.robots[0]
        robot_1 = self.robots[1]
        for idx, curr_joint_point in enumerate(orig_tensor_path[1: len(orig_tensor_path) - 1], 1):
            prev_joint_point = new_tensor_path[-1]
            next_joint_point = orig_tensor_path[idx + 1]

            # Can skip the current point in robot 0 without collision?
            prev_robot_0_point = get_robot_point_by_idx(prev_joint_point, 0)
            next_robot_0_point = get_robot_point_by_idx(next_joint_point, 0)
            robot_0_collision_free = self.single_robot_collision_free(robot_0, prev_robot_0_point, next_robot_0_point)

            # Can skip the current point in robot 1 without collision?
            prev_robot_1_point = get_robot_point_by_idx(prev_joint_point, 1)
            next_robot_1_point = get_robot_point_by_idx(next_joint_point, 1)
            robot_1_collision_free = self.single_robot_collision_free(robot_1, prev_robot_1_point, next_robot_1_point)

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

    def shorten_path(self, path):
        return self.merge_path_points(self.remove_path_points(path))
