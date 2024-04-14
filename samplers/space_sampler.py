import random

from discopygal.bindings import Point_d
from discopygal.geometry_utils import conversions
from discopygal.solvers.nearest_neighbors import NearestNeighbors_sklearn
from discopygal.solvers.metrics import Metric_Euclidean
from samplers.basic_sampler import BasicSquaresSampler


class SpaceSampler(BasicSquaresSampler):
    def __init__(self, scene, bounding_box):
        super().__init__(scene, bounding_box)

    def sample_free(self, nodes: list) -> Point_d:
        """
        Sample a free random sample for both robot 1 and robot 2 combined for robots with equal size
        """
        nearest = NearestNeighbors_sklearn()
        nearest.fit(nodes)
        # Sampling Combined
        p_rand = []
        free_positions = []
        while len(free_positions) < 2:
            sample = self.sampler.sample()
            neighbor = nearest.k_nearest(sample, 1)
            if neighbor == sample:
                continue
            if Metric_Euclidean.dist(neighbor, neighbor).to_double() < 1:
                continue
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
