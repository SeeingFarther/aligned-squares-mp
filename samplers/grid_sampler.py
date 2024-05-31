import random
import networkx as nx
import numpy as np

from discopygal.solvers import Scene
from discopygal.solvers.nearest_neighbors import NearestNeighbors_sklearn
from discopygal.bindings import FT, Point_2

from samplers.basic_sampler import BasicSquaresSampler
from utils.utils import out_of_bounds


class GridSampler(BasicSquaresSampler):
    """
    Space sampler for sampling points in a scene
    """

    def __init__(self, scene: Scene = None):
        """
        Constructor for the SpaceSampler
        :param scene:
        :type scene: :class:`~discopygal.solvers.Scene`
        """
        super().__init__(scene)

        # Initialize the space sampler
        self.roadmap = nx.Graph()
        self.nearest = NearestNeighbors_sklearn()  # remember scene bounds
        self.num_of_samples = None
        self.resolution = None
        self.space_dist = None

        if scene is None:
            self.min_x, self.max_x, self.min_y, self.max_y = None, None, None, None  # remember scene bounds
            return

        self.set_scene(scene)

    def set_scene(self, scene: Scene, bounding_box=None):
        """
        Set the scene the sampler should use

        :param scene:
        :type scene: :class:`~discopygal.solvers.Scene`
        :param bounding_box:
        :type :class:`~discopygal.Bounding_Box
        """

        super().set_scene(scene, bounding_box)

        # Set space distance
        x_dist = self.max_x - self.min_x
        y_dist = self.max_y - self.min_y
        self.gap_x = x_dist.to_double()
        self.gap_y = y_dist.to_double()

    def set_num_samples(self, num_samples: int):
        """
        Set the number of samples to be used in the space sampler
        :param num_samples:
        :type num_samples: int
        """
        self.num_of_samples = num_samples
        power_x = self.gap_x / (self.gap_x + self.gap_y)
        power_y = self.gap_y / (self.gap_x + self.gap_y)
        self.num_of_landmarks_x = int(np.ceil(np.power(num_samples, power_x)))
        self.num_of_landmarks_y = int(np.ceil(np.power(num_samples, power_y)))

    def compute_resolution(self):
        """
        Compute the resolution of the space sampler for axis x and y
        """
        self.resolution_x = int(np.ceil(self.gap_x / (self.num_of_landmarks_x+ 0.000001)))
        self.resolution_y = int(np.ceil(self.gap_y / (self.num_of_landmarks_y+ 0.000001)))

    def ready_sampler(self):
        """
        Prepare the sampler for sampling points
        """

        self.num_sample = 0
        self.compute_resolution()
        self.list_of_samples_robots = [[]] * len(self.robots)
        self.samples_visited = [{}] * len(self.robots)
        self.number_of_samples = [0] * len(self.robots)

        # Create a list of samples for each robot in the scene in grid manner
        min_x = int(self.min_x.to_double())
        max_x = int(self.max_x.to_double())
        min_y = int(self.min_y.to_double())
        max_y = int(self.max_y.to_double())
        for i in range(min_y, max_y + 1, self.resolution_y):
            for j in range(min_x, max_x + 1, self.resolution_x):

                # Check if the point is valid for each robot

                for k, robot in enumerate(self.robots):
                    # Get the robot length
                    robot_length = self.robot_lengths[k]

                    # Check different square positions for the robot in that point
                    points = [(i, j), (i - robot_length, j), (i, j - robot_length),
                              (i - robot_length, j - robot_length)]

                    # Check if point is valid for robot
                    for x_p, y_p in points:
                        sample = Point_2(FT(x_p), FT(y_p))
                        square = [(x_p, y_p), (x_p + robot_length, y_p), (x_p, y_p + robot_length), (x_p + robot_length, y_p + robot_length)]
                        if out_of_bounds(self.min_x,self.max_x, self.min_y, self.max_y, square):
                            continue

                        if self.collision_detection[self.robots[k]].is_point_valid(sample):
                            self.list_of_samples_robots[k].append(sample)
                            break

    def uniform_sample(self, robot_index: int) -> Point_2:
        """
        Sample a random uniform sample for a robot
        :param robot_index:
        :type robot_index: int

        :return: random uniform sample
        :rtype: :class:`~discopygal.bindings.Point_2`
        """
        robot = self.robots[robot_index]
        sample = super().sample()
        while not self.collision_detection[robot].is_point_valid(sample):
             sample = super().sample()

        return sample

    def sample_free(self, robot_index: int) -> Point_2:
        """
        Sample a free random grid sample for both robot 1 and robot 2 combined for robots with equal size
        :param robot_index:
        :type robot_index: int

        :return: grid sample for robots
        :rtype: :class:`~discopygal.bindings.Point_d`
        """
        # Number of free samples for the robot in the grid
        robot_samples_num = len(self.list_of_samples_robots[robot_index])

        # Check if the number of samples is reached from the list of samples
        if self.number_of_samples[robot_index] >= robot_samples_num:
            return self.uniform_sample(robot_index)

        # Choose a random sample from the list of samples
        random_index = random.randint(0, robot_samples_num - 1)
        while self.samples_visited[robot_index].get(random_index, False):
            random_index = random.randint(0, robot_samples_num - 1)
        p_rand = self.list_of_samples_robots[robot_index][random_index]
        self.number_of_samples[robot_index] += 1

        # Return point
        return p_rand
