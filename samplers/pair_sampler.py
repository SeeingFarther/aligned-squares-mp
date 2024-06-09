import random

from discopygal.bindings import Point_d, Point_2, FT
from discopygal.geometry_utils import conversions
from discopygal.solvers import Scene

from samplers.basic_sampler import BasicSquaresSampler
from utils.utils import squares_overlap, find_square_corners, compute_y_intersections, compute_x_intersections


class PairSampler(BasicSquaresSampler):
    def __init__(self, scene: Scene = None):
        """
        Constructor for the MiddleSampler.

        :param scene:
        :type scene: :class:`~discopygal.solvers.Scene`
        """

        super().__init__(scene)

        if scene is None:
            return

        # Same length for both robots? different algorithm
        self.sample_free = self.sample_free_identical if self.robot_lengths[0] == self.robot_lengths[
            1] else self.sample_free_unidentical

    def set_scene(self, scene: Scene, bounding_box=None):
        """
        Set the scene the sampler should use.
        Can be overridded to add additional processing.

        :param bounding_box:
        :type :class:`~discopygal.Bounding_Box
        :param scene: a scene to sample in
        :type scene: :class:`~discopygal.solvers.Scene`
        """

        super().set_scene(scene)
        self.sample_free = self.sample_free_identical if self.robot_lengths[0] == self.robot_lengths[
            1] else self.sample_free_unidentical

    def sample_free(self) -> Point_d:
        """
        Sample a free random sample for both robot 1 and robot 2 combined

        :return: sample
        :rtype: :class:`~discopygal.bindings.Point_d`
        """
        return self.sample_free()

    def sample_free_identical(self) -> Point_d:
        """
        Sample a free random sample for both robot 1 and robot 2 combined for robots with equal edge size

        :return: sample for two identical robots
        :rtype: :class:`~discopygal.bindings.Point_d`
        """
        # Sampling Combined
        p_rand = []
        free_positions = []
        while len(free_positions) < 2:
            sample = self.sample()

            # Try to find trivial position we can position both squares in the sampled point.
            free_positions = self.find_trivial_positions(sample)

            # If we can't find trivial positions, try to find non-trivial positions
            if len(free_positions) < 2:
                free_positions = self.find_non_trivial_x_positions(sample)
                free_positions += self.find_non_trivial_y_positions(sample)

        # Choose two free positions randomly
        i = random.randint(0, len(free_positions) - 1)
        j = i
        while j == i:
            j = random.randint(0, len(free_positions) - 1)
        p_rand.append(free_positions[i])
        p_rand.append(free_positions[j])
        p_rand = conversions.Point_2_list_to_Point_d(p_rand)

        return p_rand

    def sample_free_unidentical(self) -> Point_d:
        """
        Sample a free random sample for both robot 1 and robot 2 combined for robots with different edge size

        :return: sample for two non-identical robots
        :rtype: :class:`~discopygal.bindings.Point_d`
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

            # Create combined free positions for both robots
            for i, robot0_position in enumerate(free_positions_robot0):
                for j, robot1_position in enumerate(free_positions_robot1):
                    if self.robots_overlap(robot0_position, robot1_position):
                        continue
                    p_rand.append([robot0_position, robot1_position])

        # Choose a random pair of free positions
        i = random.randint(0, len(p_rand) - 1)
        p_rand = conversions.Point_2_list_to_Point_d(p_rand[i])
        return p_rand

    def robots_overlap(self, robot0: Point_2, robot1: Point_2) -> bool:
        """
        Check if two squares robots overlap
        :param robot0: position of robot 0
        :type robot0: :class:`~discopygal.bindings.Point_2`
        :param robot1: position of robot 1
        :type robot1: :class:`~discopygal.bindings.Point_2`

        :return: True if the robots overlap, False otherwise
        :rtype: bool
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

    def find_trivial_positions(self, square_center: Point_2, robot_index: int = 0) -> list:
        """
        Find the free positions the robot can be placed inside the square
        :param square_center: center of the square
        :type square_center: :class:`~discopygal.bindings.Point_2`
        :param robot_index: index of the robot
        :type robot_index: int

        :return list of free trivial positions for robot inside the square:
        :rtype list:
        """

        # Find corners of the square robot with edge length of sum of both robots lengths
        center_x = square_center.x().to_double()
        center_y = square_center.y().to_double()
        corners = find_square_corners(self.square_length, center_x, center_y)

        # Find positions inside the square that are collision free
        free_positions = []
        diff = self.robot_lengths[robot_index] * 0.5
        for corner in corners:
            position_x = corner[0] + diff if corner[0] < center_x else corner[0] - diff
            position_y = corner[1] + diff if corner[1] < center_y else corner[1] - diff
            position_x -= diff
            position_y -= diff
            position = Point_2(FT(position_x), FT(position_y))
            if self.collision_detection[self.robots[robot_index]].is_point_valid(position):
                free_positions.append(position)

        return free_positions

    def find_non_trivial_y_positions(self, square_center: Point_2, robot_index: int = 0) -> list:
        """
        Find the free positions the robot can be placed inside the square
        :param square_center: center of the square
        :type square_center: :class:`~discopygal.bindings.Point_2`
        :param robot_index: index of the robot
        :type robot_index: int

        :return list of free trivial positions for robot inside the square:
        :rtype list:
        """

        # Get square center
        center_x = square_center.x().to_double()
        center_y = square_center.y().to_double()

        # Find limits
        diff = self.square_length * 0.5
        min_y = center_y - diff
        max_y = center_y + diff
        min_x = center_x - diff
        max_x = center_x + diff

        # Find positions inside the square that are collision free and not trivial
        diff = self.robot_lengths[robot_index] * 0.5
        free_positions = []
        x = min_x + diff
        y_intersections = compute_y_intersections(x, min_y, max_y, self.obstacles)
        y_intersections += [min_y, max_y]
        y_intersections = list(set(y_intersections))
        free_positions += self.gap_finder.find_gap_positions_y(x, y_intersections, robot_index)

        x = max_x - diff
        y_intersections = compute_y_intersections(x, min_y, max_y, self.obstacles)
        y_intersections += [min_y, max_y]
        y_intersections = list(set(y_intersections))
        free_positions += self.gap_finder.find_gap_positions_y(x, y_intersections, robot_index)

        return free_positions

    def find_non_trivial_x_positions(self, square_center: Point_2, robot_index: int = 0):
        """
        Find the free positions the robot can be placed inside the square
        :param square_center: center of the square
        :type square_center: :class:`~discopygal.bindings.Point_2`
        :param robot_index: index of the robot
        :type robot_index: int

        :return list of free trivial positions for robot inside the square:
        :rtype list:
        """

        # Find corners
        center_x = square_center.x().to_double()
        center_y = square_center.y().to_double()

        # Find limits
        diff = self.square_length * 0.5
        min_y = center_y - diff
        max_y = center_y + diff
        min_x = center_x - diff
        max_x = center_x + diff

        # Find positions inside the square that are collision free
        diff = self.robot_lengths[robot_index] * 0.5
        free_positions = []
        y = min_y + diff
        x_intersections = compute_x_intersections(y, min_x, max_x, self.obstacles)
        x_intersections += [min_x, max_x]
        x_intersections = list(set(x_intersections))
        free_positions += self.gap_finder.find_gap_positions_x(y, x_intersections, robot_index)

        # Find positions inside the square that are collision free
        y = max_y - diff
        x_intersections = compute_x_intersections(y, min_x, max_x, self.obstacles)
        x_intersections += [min_x, max_x]
        x_intersections = list(set(x_intersections))
        free_positions += self.gap_finder.find_gap_positions_x(y, x_intersections, robot_index)

        return free_positions
