from discopygal.bindings import Point_2, FT, Polygon_2
from discopygal.geometry_utils.conversions import Point_2_to_xy
from discopygal.solvers import Scene

from samplers.basic_sampler import BasicSquaresSampler
from utils.utils import out_of_bounds, find_y_coordinate, point_inside_polygon


class MiddleSampler(BasicSquaresSampler):
    def __init__(self, scene: Scene = None):
        super().__init__(scene)

        if scene is None:
            self.min_x, self.max_x, self.min_y, self.max_y = None, None, None, None  # remember scene bounds
            return

        self.set_scene(scene)

    def compute_middle_point(self, point: Point_2, min_y: float, max_y: float, obstacles) -> float:
        p_x, p_y = Point_2_to_xy(point)
        y_top = max_y

        for obstacle in obstacles:
            obstacle: Polygon_2 = obstacle.poly

            # Inside the obstacle?
            if point_inside_polygon(p_x, p_y, obstacle):
                continue

            edges = obstacle.edges()  # each edge is a Segment_2 object.

            # Print the coordinates of each edge
            for edge in edges:
                start: Point_2 = edge.source()
                target: Point_2 = edge.target()

                if start.x() <= p_x <= target.x() or target.x() <= p_x <= start.x():
                    y_edge = find_y_coordinate(start, target, p_x, min_y, max_y)

                    if y_edge and p_y <= y_edge[0] <= y_top:  # if y == 0 then conditioning on y returns False.
                        y_top = y_edge[0]

        return (y_top + p_y) / 2

    def find_middle(self, sample):
        # Sample a point from the gaussian distribution.
        x = sample.x().to_double()
        y = self.compute_middle_point(sample, self.min_y, self.max_y, self.obstacles)
        sample_tag = Point_2(FT(x), FT(y))
        return sample_tag

    def sample_middle(self, index):
        """
        Sample in a middle axis strategy.
        """
        # The same as the pseudocode in the paper
        while True:
            sample = self.sample()
            #sample = Point_2(FT(-48.0), FT(-50.5))
            robot = self.scene.robots[index]
            robot_length = self.robot_lengths[index]

            if self.collision_detection[robot].is_point_valid(sample):
                continue

            sample_tag = self.find_middle(sample)
            if self.collision_detection[robot].is_point_valid(sample_tag):
                x, y = Point_2_to_xy(sample_tag)

                points = [(x, y), (x - robot_length, y), (x, y - robot_length), (x - robot_length, y - robot_length)]
                for x_p, y_p in points:
                    p = Point_2(FT(x_p), FT(y_p))
                    square = [(x_p, y_p), (x_p + robot_length, y_p), (x_p, y_p + robot_length),
                              (x_p + robot_length, y_p + robot_length)]
                    if out_of_bounds(self.min_x, self.max_x, self.min_y, self.max_y, square):
                        continue

                    if self.collision_detection[robot].is_point_valid(p):
                        return p

    def sample_free(self, robot_index):
        p_rand = self.sample_middle(robot_index)
        return p_rand
