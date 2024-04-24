import random

from discopygal.bindings import Point_d
from discopygal.geometry_utils import conversions
from discopygal.solvers import Scene

from samplers.basic_sampler import BasicSquaresSampler


class CombinedSquaresSampler(BasicSquaresSampler):
    def __init__(self, scene: Scene = None):
        super().__init__(scene)

    def sample_free(self) -> Point_d:
        """
        Sample a free random sample for both robot 1 and robot 2 combined for robots with equal size
        """
        # Sampling Combined
        p_rand = []
        free_positions = []
        while len(free_positions) < 2:
            sample = self.sample()

            #sample = Point_2(FT(0.0), FT(-1.0))
            free_positions = self.find_trivial_positions(sample, 0)

            if len(free_positions) < 2:
                free_positions = self.find_non_trivial_x_positions(sample, 0)

            if len(free_positions) < 2:
                free_positions = self.find_non_trivial_y_positions(sample, 0)

        # Choose free positions randomly
        i = random.randint(0, len(free_positions) - 1)
        j = i
        while j == i:
            j = random.randint(0, len(free_positions) - 1)
        p_rand.append(free_positions[i])
        p_rand.append(free_positions[j])
        p_rand = conversions.Point_2_list_to_Point_d(p_rand)
        return p_rand
