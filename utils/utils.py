import numpy as np
from shapely.geometry import Point, Polygon, LineString

from discopygal.bindings import *
from discopygal.geometry_utils.conversions import Point_2_to_xy, Polygon_2_to_array_of_points


def find_square_corners(square_length: float, center_x: float, center_y: float) -> list[list]:
    """
    Find the corners of square using it center
    :param square_length:
    :type square_length: float
    :param center_x:
    :type center_x: float
    :param center_y:
    :type center_y: float

    :return list of square corners:
    :rtype list:
    """
    diff = square_length / 2
    return [[center_x - diff, center_y + diff], [center_x + diff, center_y + diff],
            [center_x + diff, center_y - diff], [center_x - diff, center_y - diff]]


def find_x_coordinate(p1: Point_2, p2: Point_2, y: float, min_x: float, max_x: float) -> list[
    float]:
    """
    Find the x-coordinate of the intersection of the line between p1 and p2 with the y-coordinate
    :param p1:
    :type :class:`~discopygal.bindings.Point_2`
    :param p2:
    :type :class:`~discopygal.bindings.Point_2`
    :param y:
    :type y: float
    :param min_x:
    :type min_x: float
    :param max_x:
    :type max_x: float

    :return: list of x-coordinates
    :rtype: list
    """
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


def find_x_coordinate_minimal(p1: Point_2, p2: Point_2, y: float) -> float:
    """
    Find the minimal x-coordinate of the intersection of the line between p1 and p2 with the y-coordinate
    :param p1:
    :type :class:`~discopygal.bindings.Point_2`
    :param p2:
    :type :class:`~discopygal.bindings.Point_2`
    :param y:
    :type y: float

    :return: x coordinate of the intersection if we run vertical line from y to the edge
    :rtype: float
    """
    # Extracting coordinates
    x1, y1 = Point_2_to_xy(p1)
    x2, y2 = Point_2_to_xy(p2)

    if y1 == y2:
        return min(x1, x2)

    # Calculate the slope (m)
    if x2 != x1:  # Horizontal
        m = (y2 - y1) / (x2 - x1)
    else:
        return x1

    # Calculate the y-intercept (b)
    b = y1 - m * x1

    # Calculate x-coordinate for the given x-coordinate
    x_edge = (y - b) / m
    return x_edge


def find_y_coordinate_minimal(p1: Point_2, p2: Point_2, x: float) -> float:
    """
    Find the y-coordinate of the intersection of the line between p1 and p2 with the x-coordinate
    :param p1:
    :type :class:`~discopygal.bindings.Point_2`
    :param p2:
    :type :class:`~discopygal.bindings.Point_2`
    :param x:
    :type x: float

    :return: l y coordinate of the intersection if we run horizontal line from x to the edge
    :rtype: list
    """
    # Extracting coordinates
    x1, y1 = Point_2_to_xy(p1)
    x2, y2 = Point_2_to_xy(p2)

    # Calculate the slope (m)
    if x2 != x1:  # Vertical
        m = (y2 - y1) / (x2 - x1)
    else:
        return min(y1, y2)

    # Calculate the y-intercept (b)
    b = y1 - m * x1

    # Calculate y-coordinate for the given x-coordinate
    y_edge = m * x + b
    return y_edge


def find_y_coordinate(p1: Point_2, p2: Point_2, x: float, min_y: float, max_y: float) -> list[
    float]:
    """
    Find the y-coordinate of the intersection of the line between p1 and p2 with the x-coordinate
    :param p1:
    :type :class:`~discopygal.bindings.Point_2`
    :param p2:
    :type :class:`~discopygal.bindings.Point_2`
    :param x:
    :type x: float
    :param min_y:
    :type min_y: float
    :param max_y:
    :type max_y: float

    :return: list of y-coordinates
    :rtype: list
    """
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
    """
    Compute the y-intersections of a vertical line with the obstacles
    :param p_x:
    :type :class:`~discopygal.bindings.FT`
    :param min_y:
    :type min_y: float
    :param max_y:
    :type max_y: float
    :param obstacles:
    :type obstacles: list[:class:`~discopygal.bindings.Obstacle`]

    :return: list of y-intersections
    :rtype: list
    """
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
    """
    Compute the x-intersections of a horizontal line with the obstacles
    :param p_y:
    :type p_y: float
    :param min_x:
    :type min_x: float
    :param max_x:
    :type max_x: float
    :param obstacles:
    :type obstacles: list[:class:`~discopygal.bindings.Obstacle`]

    :return: list of x-intersections
    :rtype: list
    """
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


def euclidean_distance_1d(point1: float, point2: float) -> float:
    """
    Compute the euclidean distance between two points in 1D
    :param point1:
    :type point1: float
    :param point2:
    :type point2: float

    :return: distance
    :rtype: float
    """
    # Compute absolute difference between coordinates
    distance = abs(point2 - point1)
    return distance


def inside_limits(x: float, value1: float, value2: float) -> bool:
    """
    Check if x is inside the limits of value1 and value2
    :param x:
    :type x: float
    :param value1:
    :type value1: float
    :param value2:
    :type value2: float

    :return: True if x is inside the limits, False otherwise
    :rtype: bool
    """
    min_value = min(value1, value2)
    max_value = max(value1, value2)
    return min_value <= x <= max_value


def inside_limits_fast(x: float, min_value: float, max_value: float) -> bool:
    """
    Check if x is inside the limits of min_value and max_value
    :param x:
    :type x: float
    :param min_value:
    :type min_value: float
    :param max_value:
    :type max_value: float

    :return: True if x is inside the limits, False otherwise
    :rtype: bool
    """
    return min_value <= x <= max_value


def find_max_value_coordinates(arr: list) -> tuple:
    """
    Find the coordinates of the max value in the array
    :param arr:
    :type arr: list

    :return: tuple of index and value
    :rtype: tuple
    """
    # return the index of the max cell in the array
    arr = np.array(arr)
    max_index = np.unravel_index(np.argmax(arr, axis=None), arr.shape)
    return max_index


def get_point_d(robot_idx_to_shorten: int, prev_next_idx_to_shorten: int, prev_joint_point: Point_d,
                orig_curr_joint_point: Point_d, next_joint_point: Point_d) -> Point_d:
    """
    Get the point_d for the robot to shorten
    :param robot_idx_to_shorten:
    :type robot_idx_to_shorten: int
    :param prev_next_idx_to_shorten:
    :type prev_next_idx_to_shorten: int
    :param prev_joint_point:
    :type prev_joint_point: Point_d
    :param orig_curr_joint_point
    :type orig_curr_joint_point: Point_d
    :param next_joint_point:
    :type next_joint_point: Point_d

    :return: Point_d
    :rtype: Point_d
    """
    result = [0, 0, 0, 0]
    new_point = prev_joint_point if prev_next_idx_to_shorten == 0 else next_joint_point
    result[robot_idx_to_shorten * 2] = new_point[robot_idx_to_shorten * 2]
    result[robot_idx_to_shorten * 2 + 1] = new_point[robot_idx_to_shorten * 2 + 1]
    other_robot_idx = 1 - robot_idx_to_shorten
    result[other_robot_idx * 2] = orig_curr_joint_point[other_robot_idx * 2]
    result[other_robot_idx * 2 + 1] = orig_curr_joint_point[other_robot_idx * 2 + 1]
    return Point_d(4, result)


def get_robot_point_by_idx(point_d: Point_d, robot_idx: int) -> Point_2:
    """
    Get the point of the robot by index
    :param point_d:
    :type point_d: Point_d
    :param robot_idx:
    :type robot_idx: int

    :return: Point_2 of the robot corresponding to the index
    :rtype: Point_2
    """
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


def point_inside_square(point: tuple, square: list) -> bool:
    """
    Check if a point is inside a square
    :param point:
    :type point: tuple
    :param square:
    :type square: list

    :return: True if the point is inside the square, False otherwise
    :rtype: bool
    """
    x, y = point
    x1, y1, x2, y2 = get_square_coordinates(square)
    return x1 < x < x2 and y1 < y < y2


def squares_overlap(square1: list, square2: list) -> bool:
    """
    Check if two squares overlap
    :param square1:
    :type square1: list
    :param square2:
    :type square2: list

    :return: True if the squares overlap, False otherwise
    :rtype: bool
    """
    for point in square1:
        if point_inside_square(point, square2):
            return True
    for point in square2:
        if point_inside_square(point, square1):
            return True
    return False


def out_of_bounds(x_min: float, x_max: float, y_min: float, y_max: float, square: list) -> bool:
    """
    Check if a square is out of bounds
    :param x_min:
    :type x_min: float
    :param x_max:
    :type x_max: float
    :param y_min:
    :type y_min: float
    :param y_max:
    :type y_max: float
    :param square:
    :type square: list

    :return: True if the square is out of bounds, False otherwise
    :rtype: bool
    """
    x1, y1, x2, y2 = get_square_coordinates(square)
    if x1 < x_min or x2 > x_max or y1 < y_min or y2 > y_max:
        return True
    return False


def point_inside_polygon(x: float, y: float, poly: Polygon_2) -> bool:
    """
    Check if a point is inside a polygon
    :param x:
    :type x: float
    :param y:
    :type y: float
    :param poly:
    :type poly: Polygon_2

    :return: True if the point is inside the polygon, False otherwise
    :rtype: bool
    """

    point = Point(x, y)
    vertices = Polygon_2_to_array_of_points(poly)
    polygon = Polygon(vertices)
    return polygon.contains(point)


def line_inside_polygon(x1: float, y1: float, x2: float, y2: float, poly: Polygon_2) -> bool:
    """
    Check if a line is inside a polygon
    :param x1:
    :type x1: float
    :param y1:
    :type y1: float
    :param x2:
    :type x2: float
    :param y2:
    :type y2: float
    :param poly:
    :type poly: Polygon_2

    :return: check if the line is inside the polygon
    :rtype: bool
    """
    line = LineString([[x1, y1], [x2, y2]])
    vertices = Polygon_2_to_array_of_points(poly)
    polygon = Polygon(vertices)
    if not polygon.contains(line):
        return False
    return True


def point2_to_point_d(point2: Point_2) -> Point_d:
    """
    Convert a Point_2 to a Point_d
    :param point2:
    :type point2: Point_2

    :return: Point_d
    :rtype: Point_d
    """

    return Point_d(2, [point2.x().to_double(), point2.y().to_double()])
