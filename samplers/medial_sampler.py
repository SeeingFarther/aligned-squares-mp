import random

import numpy as np
from discopygal.bindings import Point_2, FT, Polygon_2
from discopygal.geometry_utils.conversions import Point_2_to_xy
from discopygal.solvers import Scene

from samplers.basic_sampler import BasicSquaresSampler
from utils.utils import out_of_bounds, line_inside_polygon, find_x_coordinate_minimal, find_y_coordinate_minimal, \
    point_inside_obstacle


class MedialSampler(BasicSquaresSampler):
    """
    Middle sampler for sampling points in a scene.
    """

    def __init__(self, scene: Scene = None, y_axis: bool = True):
        """
        Constructor for the MiddleSampler.

        :param scene:
        :type scene: :class:`~discopygal.solvers.Scene`
        """
        super().__init__(scene)
        self.find_middle = self.find_middle_y if y_axis else self.find_middle_x

        if scene is None:
            self.min_x, self.max_x, self.min_y, self.max_y = None, None, None, None  # remember scene bounds
            return

        self.set_scene(scene)

    def compute_middle_point_x(self, point: Point_2, obstacles: list) -> float:
        """
        Compute the middle point of the point and the closet upward obstacle edge in the scene in the x-axis.
        :param point:
        :type point: :class:`~discopygal.bindings.Point_2`
        :param obstacles:
        :type obstacles: list[:class:`~discopygal.bindings.Obstacle`]

        :return: middle point in the x-axis with the closest upward obstacle edge in the x-axis
        :rtype: float
        """
        p_x, p_y = Point_2_to_xy(point)
        x_top = np.inf

        for obstacle in obstacles:
            obstacle: Polygon_2 = obstacle.poly
            edges = obstacle.edges()

            # Get the x-intersections of the horizontal line with the obstacles edges
            for edge in edges:
                start: Point_2 = edge.source()
                target: Point_2 = edge.target()

                # Check if the horizontal line intersects with the edge from p_x
                if p_x == target.x() or p_y == start.x():
                    continue

                if start.y() <= p_y <= target.y() or target.y() <= p_y <= start.y():
                    x_edge = find_x_coordinate_minimal(start, target, p_y)

                    # Check if the x-intersection is closer to the point than the current x_top
                    if p_x <= x_edge <= x_top and (not line_inside_polygon(p_x, p_y, x_edge, p_y,
                                                                                        obstacle)):
                        x_top = x_edge

        return (x_top + p_x) / 2

    def find_middle_x(self, sample: Point_2) -> Point_2:
        """
        Find the middle point of the sample.
        :param sample:
        :type sample: :class:`~discopygal.bindings.Point_2`

        :return: middle point of the sample in x-axis
        :rtype: :class:`~discopygal.bindings.Point_2`
        """

        # Find middle point of the sample and the closest upward obstacle edge in the y-axis
        x = self.compute_middle_point_x(sample, self.obstacles)
        if x == np.inf:
            return None

        # Return the middle point
        sample_tag = Point_2(FT(x), sample.y())
        return sample_tag

    def compute_middle_point_y(self, point: Point_2, obstacles: list) -> float:
        """
        Compute the middle point of the point and the closet upward obstacle edge in the scene in the y-axis.
        :param point:
        :type point: :class:`~discopygal.bindings.Point_2`
        :param obstacles:
        :type obstacles: list[:class:`~discopygal.bindings.Obstacle`]

        :return: middle point in the y-axis with the closest upward obstacle edge in the y-axis
        :rtype: float
        """
        p_x, p_y = Point_2_to_xy(point)
        y_top = np.inf

        for obstacle in obstacles:
            obstacle: Polygon_2 = obstacle.poly
            edges = obstacle.edges()

            # Get the y-intersections of the vertical line with the obstacles edges
            for edge in edges:
                start: Point_2 = edge.source()
                target: Point_2 = edge.target()

                # Check if the vertical line intersects with the edge from p_y
                if start.x() <= p_x <= target.x() or target.x() <= p_x <= start.x():
                    y_edge = find_y_coordinate_minimal(start, target, p_x)

                    # Check if the y-intersection is closer to the point than the current y_top
                    if p_y <= y_edge <= y_top and (not line_inside_polygon(p_x, p_y, p_x, y_edge,
                                                                                        obstacle)):
                        y_top = y_edge

        return (y_top + p_y) / 2

    def find_middle_y(self, sample: Point_2) -> Point_2:
        """
        Find the middle point of the sample.
        :param sample:
        :type sample: :class:`~discopygal.bindings.Point_2`

        :return: middle point of the sample in y-axis
        :rtype: :class:`~discopygal.bindings.Point_2`
        """

        # Find middle point of the sample and the closest upward obstacle edge in the y-axis
        y = self.compute_middle_point_y(sample, self.obstacles)
        if y == np.inf:
            return None

        # Return the middle point
        sample_tag = Point_2(sample.x(), FT(y))
        return sample_tag

    def sample_middle(self, robot_index: int) -> Point_2:
        """
        Sample in a middle axis strategy.
        :param robot_index:
        :type robot_index: int

        :return: sample
        :rtype: :class:`~discopygal.bindings.Point_2`
        """
        while True:

            # Sample a point
            sample = self.sample()

            # Get the robot and its length
            robot = self.scene.robots[robot_index]
            robot_length = self.robot_lengths[robot_index]

            # If the point is valid? sample a new point
            x, y = Point_2_to_xy(sample)
            if not point_inside_obstacle(x, y, self.obstacles):
                continue

            # Get middle point
            middle_point = self.find_middle(sample)
            if middle_point is None:
                continue

            # Check different square positions for the robot in that point
            x, y = Point_2_to_xy(middle_point)
            points = [(x, y), (x - robot_length, y), (x, y - robot_length), (x - robot_length, y - robot_length)]
            random.shuffle(points)

            # Check if the square is out of bounds
            for x_p, y_p in points:

                # Square is out of bounds? we can skip it
                p = Point_2(FT(x_p), FT(y_p))
                square = [(x_p, y_p), (x_p + robot_length, y_p), (x_p, y_p + robot_length),
                          (x_p + robot_length, y_p + robot_length)]
                if out_of_bounds(self.min_x, self.max_x, self.min_y, self.max_y, square):
                    continue

                # Point valid?
                if self.collision_detection[robot].is_point_valid(p):
                    return p

    def sample_free(self, robot_index: int) -> Point_2:
        """
        Sample point
        :param robot_index:
        :type: int

        :return: sample
        :rtype: :class:`~discopygal.bindings.Point_2`
        """
        p_rand = self.sample_middle(robot_index)
        return p_rand
