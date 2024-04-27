import random
import networkx as nx
import numpy as np

from discopygal.geometry_utils import conversions
from discopygal.solvers import Scene
from discopygal.solvers.nearest_neighbors import NearestNeighbors_sklearn
from discopygal.bindings import FT, Point_2, Point_d

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
        self.gap_x = x_dist.to_double()
        self.gap_y = y_dist.to_double()

    def set_num_samples(self, num_samples: int):
        self.num_of_samples = num_samples
        power_x = self.gap_x / (self.gap_x + self.gap_y)
        power_y = self.gap_y / (self.gap_x + self.gap_y)
        self.num_of_landmarks_x = int(np.ceil(np.power(num_samples, power_x)))
        self.num_of_landmarks_y = int(np.ceil(np.power(num_samples, power_y)))

    def compute_resolution(self):
        self.resolution_x = int(np.ceil(self.gap_x / self.num_of_landmarks_x))
        self.resolution_y = int(np.ceil(self.gap_y / self.num_of_landmarks_y))

    def ready_sampler(self):
        self.num_sample = 0
        self.compute_resolution()
        self.list_of_samples_robots = [[], []]
        self.samples_visited = [{}, {}]

        min_x = int(self.min_x.to_double())
        max_x = int(self.max_x.to_double())
        min_y = int(self.min_y.to_double())
        max_y = int(self.max_y.to_double())
        for i in range(min_y, max_y + 1, self.resolution_y):
            for j in range(min_x, max_x + 1, self.resolution_x):
                sample = Point_2(FT(j), FT(i))
                if self.collision_detection[self.robots[0]].is_point_valid(sample):
                    self.list_of_samples_robots[0].append(sample)

                if self.collision_detection[self.robots[1]].is_point_valid(sample):
                    self.list_of_samples_robots[1].append(sample)

    def uniform_sample(self) -> Point_d:
        p_rand = []
        for robot in self.scene.robots:
            sample = super().sample()
            while not self.collision_detection[robot].is_point_valid(sample):
                sample = super().sample()
            p_rand.append(sample)
        p_rand = conversions.Point_2_list_to_Point_d(p_rand)
        return p_rand

    def sample_free(self, robot_index) -> Point_d:
        """
        Sample a free random sample for both robot 1 and robot 2 combined for robots with equal size
        """
        random_index = random.randint(0, len(self.list_of_samples_robots[robot_index]) - 1)
        while self.samples_visited[robot_index].get(random_index, False):
            random_index = random.randint(0, len(self.list_of_samples_robots[robot_index]) - 1)
        p_rand = self.list_of_samples_robots[robot_index][random_index]

        # Return point
        return p_rand
