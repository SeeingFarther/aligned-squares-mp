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
    x1, y1 = p1.x().to_double(), p1.y().to_double()
    x2, y2 = p2.x().to_double(), p2.y().to_double()

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