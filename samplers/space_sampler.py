import random
import networkx as nx
import numpy as np

from discopygal.bindings import Point_d
from discopygal.geometry_utils import conversions
from discopygal.solvers import Scene
from discopygal.solvers.nearest_neighbors import NearestNeighbors_sklearn
from discopygal.solvers.metrics import Metric_Euclidean

from samplers.basic_sampler import BasicSquaresSampler


class SpaceSampler(BasicSquaresSampler):
    def __init__(self, scene: Scene = None):
        super().__init__(scene)

        self.roadmap = nx.Graph()
        self.nearest = NearestNeighbors_sklearn()  # remember scene bounds
        self.num_of_samples = None
        self.resolution = None
        self.space_dist = None

        if scene is None:
            self.min_x, self.max_x, self.min_y, self.max_y = None, None, None, None  # remember scene bounds
            return

        self.set_scene(scene)

    def set_scene(self, scene, bounding_box=None):
        super().set_scene(scene, bounding_box)

        # Set space distance
        x_dist = self.max_x - self.min_x
        y_dist = self.max_y - self.min_y
        self.gap_x = x_dist.to_double() - 1
        self.gap_y = y_dist.to_double() - 1

        # Compute resolution
        self.compute_resolution()

    def set_num_samples(self, num_samples: int):
        self.num_of_samples = num_samples
        self.num_of_landmarks = int(np.ceil(np.sqrt(num_samples)) + 1)

    def compute_resolution(self):
        self.resolution_x = self.gap_x / self.num_of_samples
        self.resolution_y = self.gap_y / self.num_of_samples

    def ready_sampler(self):
        self.roadmap.clear()
        self.num_sample = 0
        for i in range(self.num_of_landmarks):
            # Sampling
            nodes = list(self.roadmap.nodes)
            self.nearest.fit(nodes)

            p_rand = None
            while p_rand is None:

                # Sample point
                sample = self.sample()
                if not self.collision_detection[self.robots[0]].is_point_valid(sample):
                    continue

                # Find nearest neighbor
                neighbor = self.nearest.k_nearest(sample, 1)

                # Already sampled?
                if neighbor and neighbor[0] == sample:
                    continue

                # Too close?
                if neighbor and ((sample.x() - neighbor[0].x()).to_double() < self.gap_x):
                    self.gap_x -= self.resolution_x
                    continue

                # Too close?
                if neighbor and ((sample.y() - neighbor[0].y()).to_double() < self.gap_y):
                    self.gap_y -= self.resolution_y
                    continue

                # Find free positions
                p_rand = sample
            # Add to roadmap
            self.roadmap.add_node(p_rand)

        self.list_of_samples = []
        nodes = list(self.roadmap.nodes)
        for i in range(self.num_of_landmarks):
            for j in range(self.num_of_landmarks):
                if i == j:
                    continue

                p1 = nodes[i]
                p2 = nodes[j]
                sample = conversions.Point_2_list_to_Point_d([p1, p2])
                self.list_of_samples.append(sample)

    def uniform_sample(self) -> Point_d:
        p_rand = []
        for robot in self.scene.robots:
            sample = super().sample()
            while not self.collision_detection[robot].is_point_valid(sample):
                sample = super().sample()
            p_rand.append(sample)
        p_rand = conversions.Point_2_list_to_Point_d(p_rand)
        return p_rand

    def sample_free(self) -> Point_d:
        """
        Sample a free random sample for both robot 1 and robot 2 combined for robots with equal size
        """
        if self.num_sample < len(self.list_of_samples):
            p_rand = self.list_of_samples[self.num_sample]
            self.num_sample += 1
        else:
            p_rand = self.uniform_sample()

        # Return point
        return p_rand
