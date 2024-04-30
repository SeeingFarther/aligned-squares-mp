import numpy as np
from discopygal.bindings import *
from discopygal.geometry_utils.conversions import Point_2_to_xy, Polygon_2_to_array_of_points
from shapely.geometry import Point, Polygon


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
    x1, y1 = Point_2_to_xy(p1)
    x2, y2 = Point_2_to_xy(p2)

    x1_inside = inside_limits_fast(x1, min_x, max_x)
    x2_inside = inside_limits_fast(x2, min_x, max_x)
    if not x1_inside and not x2_inside:
        return []

    # Calculate the slope (m)
    if x2 != x1:  # Horizontal
        m = (y2 - y1) / (x2 - x1)
    else:
        return [x1]

    # Vertical line?
    if m == 0:
        result = []
        if x1_inside:
            result.append(x1)
        if x2_inside:
            result.append(x2)
        return result

    # Calculate the y-intercept (b)
    b = y1 - m * x1

    # Calculate y-coordinate for the given x-coordinate
    x_edge = (y - b) / m

    if min_x <= x_edge <= max_x:
        return [x_edge]
    return []


def find_y_coordinate(p1: Point_2, p2: Point_2, x: float, min_y: float, max_y: float) -> list[
    float]:
    # Extracting coordinates
    x1, y1 = Point_2_to_xy(p1)
    x2, y2 = Point_2_to_xy(p2)

    y1_inside = inside_limits_fast(y1, min_y, max_y)
    y2_inside = inside_limits_fast(y2, min_y, max_y)
    if not y1_inside and not y2_inside:
        return []

    # Calculate the slope (m)
    if x2 != x1:  # Vertical
        m = (y2 - y1) / (x2 - x1)
    else:
        if y1 == y2:
            return [y1]

        result = []
        if y2_inside:
            result.append(y2)
        if y1_inside:
            result.append(y1)
        return result

    # Calculate the y-intercept (b)
    b = y1 - m * x1

    # Calculate y-coordinate for the given x-coordinate
    y_edge = m * x + b

    if inside_limits_fast(y_edge, min_y, max_y):
        return [y_edge]
    return []


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


def euclidean_distance_1d(point1, point2):
    # Compute absolute difference between coordinates
    distance = abs(point2 - point1)
    return distance


def inside_limits(x, value1, value2):
    min_value = min(value1, value2)
    max_value = max(value1, value2)
    return min_value <= x <= max_value


def inside_limits_fast(x, min_value, max_value):
    return min_value <= x <= max_value


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


def get_square_coordinates(square):
    x_values = [point[0] for point in square]
    y_values = [point[1] for point in square]
    x1 = min(x_values)
    x2 = max(x_values)
    y1 = min(y_values)
    y2 = max(y_values)
    return x1, y1, x2, y2


def point_inside_square(point, square):
    x, y = point
    x1, y1, x2, y2 = get_square_coordinates(square)
    return x1 < x < x2 and y1 < y < y2


def squares_overlap(square1, square2):
    for point in square1:
        if point_inside_square(point, square2):
            return True
    for point in square2:
        if point_inside_square(point, square1):
            return True
    return False


def out_of_bounds(x_min, x_max, y_min, y_max, square):
    x1, y1, x2, y2 = get_square_coordinates(square)
    if x1 < x_min or x2 > x_max or y1 < y_min or y2 > y_max:
        return True
    return False


def point_inside_polygon(x, y, poly):
    point = Point(x, y)
    vertices = Polygon_2_to_array_of_points(poly)
    polygon = Polygon(vertices)
    return polygon.contains(point)


def point2_to_point_d(point2: Point_2):
    return Point_d(2, [point2.x().to_double(), point2.y().to_double()])



