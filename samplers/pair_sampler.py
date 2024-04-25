import random

from discopygal.bindings import Point_d
from discopygal.geometry_utils import conversions
from discopygal.solvers import Scene

from samplers.basic_sampler import BasicSquaresSampler
from utils.utils import squares_overlap


class PairSampler(BasicSquaresSampler):
    def __init__(self, scene: Scene = None):
        super().__init__(scene)

        if scene is None:
            return

        self.sample_free = self.sample_free_identical if self.robot_lengths[0] == self.robot_lengths[
            1] else self.sample_free_unidentical

    def set_scene(self, scene, bounding_box=None):
        """
        Set the scene the sampler should use.
        Can be overridded to add additional processing.

        :param bounding_box:
        :param num_samples:
        :param scene: a scene to sample in
        :type scene: :class:`~discopygal.solvers.Scene`
        """
        super().set_scene(scene)
        self.sample_free = self.sample_free_identical if self.robot_lengths[0] == self.robot_lengths[
            1] else self.sample_free_unidentical

    def sample_free(self) -> Point_d:
        """
        Sample a free random sample for both robot 1 and robot 2 combined
        """
        return self.sample_free()

    def sample_free_identical(self) -> Point_d:
        """
        Sample a free random sample for both robot 1 and robot 2 combined for robots with equal edge size
        """
        # Sampling Combined
        p_rand = []
        free_positions = []
        while len(free_positions) < 2:
            sample = self.sample()

            # sample = Point_2(FT(0.0), FT(-1.0))
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
        # for i in range(len(p_rand)):
        #     for j in range(len(p_rand)):
        #         if i == j:
        #             continue
        #         p = conversions.Point_2_list_to_Point_d([p_rand[i], p_rand[j]])
        #         p_rand.append(p)

        return p_rand

    def sample_free_unidentical(self) -> Point_d:
        """
        Sample a free random sample for both robot 1 and robot 2 combined for robots with different edge size
        """
        # Sampling Combined
        p_rand = []

        while not p_rand:
            sample = self.sample()

            # Find robot 0 free positions
            free_positions_robot0 = self.find_trivial_positions(sample, 0)
            free_positions_robot0 += self.find_non_trivial_x_positions(sample, 0)
            free_positions_robot0 += self.find_non_trivial_y_positions(sample, 0)

            # Find robot 1 free positions
            free_positions_robot1 = self.find_trivial_positions(sample, 1)
            free_positions_robot1 += self.find_non_trivial_x_positions(sample, 1)
            free_positions_robot1 += self.find_non_trivial_y_positions(sample, 1)

            for i, robot0_position in enumerate(free_positions_robot0):
                for j, robot1_position in enumerate(free_positions_robot1):
                    if self.robots_overlap(robot0_position, robot1_position):
                        continue
                    p_rand.append([robot0_position, robot1_position])

        i = random.randint(0, len(p_rand) - 1)
        p_rand = conversions.Point_2_list_to_Point_d(p_rand[i])
        # for i in range(len(p_rand)):
        #     p_rand[i] = conversions.Point_2_list_to_Point_d(p_rand[i])
        return p_rand

    def robots_overlap(self, robot0, robot1):
        """
        Check if two squares overlap
        """
        # Find robot 0 limits
        min_y = robot0.y().to_double()
        max_y = robot0.y().to_double() + self.robot_lengths[0]
        min_x = robot0.x().to_double()
        max_x = robot0.x().to_double() + self.robot_lengths[0]
        square0 = [(min_x, min_y), (max_x, max_y), (max_x, max_y), (max_x, min_y)]

        # Find robot 1 limits
        min_y = robot1.y().to_double()
        max_y = robot1.y().to_double() + self.robot_lengths[1]
        min_x = robot1.x().to_double()
        max_x = robot1.x().to_double() + self.robot_lengths[1]
        square1 = [(min_x, min_y), (max_x, max_y), (max_x, max_y), (max_x, min_y)]

        # Check if squares overlap
        return squares_overlap(square0, square1)
