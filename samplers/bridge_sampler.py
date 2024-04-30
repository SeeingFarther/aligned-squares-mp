import random

import numpy as np

from discopygal.bindings import Point_2, FT
from discopygal.geometry_utils.conversions import Point_2_to_xy
from discopygal.solvers import Scene

from samplers.basic_sampler import BasicSquaresSampler
from utils.utils import out_of_bounds


class BridgeSampler(BasicSquaresSampler):
    def __init__(self, scene: Scene = None):
        super().__init__(scene)
        self.std_dev_x = 0.5
        self.std_dev_y = 0.5

        if scene is None:
            self.min_x, self.max_x, self.min_y, self.max_y = None, None, None, None  # remember scene bounds
            return

        self.set_scene(scene)

    def sample_gauss(self, sample):
        # Sample a point from the gaussian distribution.
        x, y = Point_2_to_xy(sample)
        cov_matrix = [[self.std_dev_x ** 2, 0], [0, self.std_dev_y ** 2]]
        mean = [x, y]
        s = np.random.multivariate_normal(mean, cov_matrix, size=1)[0]

        # Check if out of bounding box.
        min_x = self.min_x.to_double()
        min_y = self.min_y.to_double()
        max_x = self.max_x.to_double()
        max_y = self.max_y.to_double()
        while not (min_x <= s[0] <= max_x and min_y <= s[1] <= max_y):
            s = np.random.multivariate_normal(mean, cov_matrix, size=1)[0]

        sample_tag = Point_2(FT(s[0]), FT(s[1]))
        return sample_tag

    def sample_bridge(self, index):
        """
        Sample in a bridge strategy.
        """
        # The same as the pseudocode in the paper
        while True:
            sample = self.sample()
            robot = self.scene.robots[index]
            robot_length = self.robot_lengths[index]
            if not self.collision_detection[robot].is_point_valid(sample):
                sample_tag = self.sample_gauss(sample)
                if not self.collision_detection[robot].is_point_valid(sample_tag):
                    x = (sample.x() + sample_tag.x()).to_double() / 2
                    y = (sample.y() + sample_tag.y()).to_double() / 2
                    points = [(x, y), (x - robot_length, y), (x, y - robot_length), (x - robot_length, y - robot_length)]
                    random.shuffle(points)
                    for x_p, y_p in points:
                        p = Point_2(FT(x_p), FT(y_p))
                        square = [(x_p, y_p), (x_p + robot_length, y_p), (x_p, y_p + robot_length), (x_p + robot_length, y_p + robot_length)]
                        if out_of_bounds(self.min_x,self.max_x, self.min_y, self.max_y, square):
                            continue

                        if self.collision_detection[robot].is_point_valid(p):
                            return p

    def sample_free(self, robot_index):
        p_rand = self.sample_bridge(robot_index)
        return p_rand
