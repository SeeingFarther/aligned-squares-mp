import numpy as np
from discopygal.bindings import *


def find_square_corners(square_length, center_x, center_y):
    """
    Find the corners of square using is center
    :return list of square corners:
    :rtype list:
    """
    diff = square_length / 2
    return [[center_x - diff, center_y + diff], [center_x + diff, center_y + diff],
            [center_x + diff, center_y - diff], [center_x - diff, center_y - diff]]


def find_x_coordinate(p1: Point_2, p2: Point_2, y: float, min_x: float, max_x: float) -> list[
    float]:
    # Extracting coordinates
    x1, y1 = p1.x().to_double(), p1.y().to_double()
    x2, y2 = p2.x().to_double(), p2.y().to_double()

    if not min_x <= x2 <= max_x and not min_x <= x1 <= max_x:
        return []

    # Calculate the slope (m)
    if x2 != x1:  # Horizontal
        m = (y2 - y1) / (x2 - x1)
    else:
        if x2 < x1:
            return [x2] if not min_x <= x1 <= max_x else [x2, x1]
        else:
            return [x1] if not min_x <= x1 <= max_x else [x1, x2]

    # Calculate the y-intercept (b)
    b = y1 - m * x1

    # Calculate y-coordinate for the given x-coordinate
    x_edge = (y - b) / m

    return [x_edge]


def find_y_coordinate(p1: Point_2, p2: Point_2, x: float, min_y: float, max_y: float) -> list[
    float]:
    # Extracting coordinates
    x1, y1 = p1.x().to_double(), p1.y().to_double()
    x2, y2 = p2.x().to_double(), p2.y().to_double()

    if not min_y <= y2 <= max_y and not min_y <= y1 <= max_y:
        return []

    # Calculate the slope (m)
    if x2 != x1:  # Vertical
        m = (y2 - y1) / (x2 - x1)
    else:
        if y2 < y1:
            return [y2] if not min_y <= y1 <= max_y else [y2, y1]
        else:
            return [y1] if not min_y <= y1 <= max_y else [y1, y2]

    # Calculate the y-intercept (b)
    b = y1 - m * x1

    # Calculate y-coordinate for the given x-coordinate
    y_edge = m * x + b

    return [y_edge]


def compute_y_intersections(p_x: float, min_y: float, max_y: float, obstacles) -> list:
    y_intersections = []

    for obstacle in obstacles:
        obstacle: Polygon_2 = obstacle.poly
        edges = obstacle.edges()  # each edge is a Segment_2 object.

        # Print the coordinates of each edge
        for edge in edges:
            start: Point_2 = edge.source()
            target: Point_2 = edge.target()

            if start.x() <= p_x <= target.x() or target.x() <= p_x <= start.x():
                y_intersections += find_y_coordinate(start, target, p_x, min_y, max_y)

    return y_intersections


def compute_x_intersections(p_y: float, min_x: float, max_x: float, obstacles) -> list:
    x_intersections = []

    for obstacle in obstacles:
        obstacle: Polygon_2 = obstacle.poly
        edges = obstacle.edges()  # each edge is a Segment_2 object.

        # Print the coordinates of each edge
        for edge in edges:
            start: Point_2 = edge.source()
            target: Point_2 = edge.target()

            if start.y() <= p_y <= target.y() or target.y() <= p_y <= start.y():
                x_intersections += find_x_coordinate(start, target, p_y, min_x, max_x)

    return x_intersections


def find_max_value_coordinates(arr):
    # return the index of the max cell in the array
    arr = np.array(arr)
    max_index = np.unravel_index(np.argmax(arr, axis=None), arr.shape)
    return max_index


def get_point_d(robot_idx_to_shorten: int, prev_next_idx_to_shorten: int, prev_joint_point: Point_d,
                orig_curr_joint_point: Point_d, next_joint_point: Point_d) -> Point_d:
    result = [0, 0, 0, 0]
    new_point = prev_joint_point if prev_next_idx_to_shorten == 0 else next_joint_point
    result[robot_idx_to_shorten * 2] = new_point[robot_idx_to_shorten * 2]
    result[robot_idx_to_shorten * 2 + 1] = new_point[robot_idx_to_shorten * 2 + 1]
    other_robot_idx = 1 - robot_idx_to_shorten
    result[other_robot_idx * 2] = orig_curr_joint_point[other_robot_idx * 2]
    result[other_robot_idx * 2 + 1] = orig_curr_joint_point[other_robot_idx * 2 + 1]
    return Point_d(4, result)


def get_robot_point_by_idx(point_d: Point_d, robot_idx: int):
    # utility function to get point in array
    return Point_2(point_d[robot_idx * 2], point_d[robot_idx * 2 + 1])


