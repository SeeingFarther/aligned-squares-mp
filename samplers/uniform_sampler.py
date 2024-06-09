from discopygal.bindings import *
from discopygal.geometry_utils import collision_detection
from discopygal.geometry_utils.bounding_boxes import calc_scene_bounding_box

from samplers.basic_sampler import BasicSquaresSampler


class UniformSampler(BasicSquaresSampler):
    """
    Uniform sampler in the scene

    :param scene: a scene to sample in
    :type scene: :class:`~discopygal.solvers.Scene`
    """

    def __init__(self, scene=None):
        super().__init__(scene)
        if scene is None:
            self.min_x, self.max_x, self.min_y, self.max_y = None, None, None, None  # remember scene bounds
        else:
            self.set_scene(scene)

    def set_bounds_manually(self, min_x, max_x, min_y, max_y):
        """
        Set the sampling bounds manually (instead of supplying a scene)
        Bounds are given in CGAL :class:`~discopygal.bindings.FT`
        """
        self.min_x, self.max_x, self.min_y, self.max_y = min_x, max_x, min_y, max_y

    def set_scene(self, scene, bounding_box=None):
        """
        Set the scene the sampler should use.
        Can be overridded to add additional processing.

        :param scene: a scene to sample in
        :type scene: :class:`~discopygal.solvers.Scene`
        """
        super().set_scene(scene)
        self.min_x, self.max_x, self.min_y, self.max_y = bounding_box or calc_scene_bounding_box(self.scene)

        self.obstacles = scene.obstacles
        self.collision_detection = {}

        # Build collision detection for each robot
        i = 0
        for robot in scene.robots:
            self.robots.append(robot)
            self.collision_detection[robot] = collision_detection.ObjectCollisionDetection(scene.obstacles, robot)

    def sample_free(self, robot_index: int) -> Point_2:
        """
        Return a sample in the space (might be invalid)

        :return: sampled point
        :rtype: :class:`~discopygal.bindings.Point_2`
        """
        while True:
            sample = self.sample()
            if self.collision_detection[self.robots[robot_index]].is_point_valid(sample):
                return sample
