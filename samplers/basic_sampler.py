import networkx
import random

from discopygal.geometry_utils.bounding_boxes import calc_scene_bounding_box
from discopygal.geometry_utils import collision_detection
from discopygal.solvers import Scene
from discopygal.solvers.metrics import Metric_Euclidean, Metric
from discopygal.solvers.samplers import Sampler

from utils.utils import *
from utils.gap_position_finder import GapPositionFinder


class BasicSquaresSampler(Sampler):
    """
    Basic sampler for sampling squares in a scene
    """
    def __init__(self, scene: Scene, metric: Metric = None, graph: networkx.Graph = None):
        """
        Constructor
        :param scene:
        :type scene: :class:`~discopygal.solvers.Scene`
        :param metric:
        :type metric: :class:`~discopygal.solvers.metrics.Metric`
        :param graph:
        :type graph: :class:`~networkx.Graph`
        """

        # Initialize the sampler
        super().__init__(scene)
        self.gap_finder = None
        self.square_length = None
        self.robot_lengths = [0, 0]
        self.robots = []
        self.metric = metric
        self.roadmap = graph

        if scene is None:
            return

        self.set_scene(scene)

    def set_scene(self, scene: Scene,  bounding_box = None):
        """
        Set the scene the sampler should use.
        Can be overridded to add additional processing.

        :param bounding_box:
        :type :class:`~discopygal.Bounding_Box
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

            # Get squares robots edges length
            for e in robot.poly.edges():
                self.robot_lengths[i] = Metric_Euclidean.dist(e.source(), e.target()).to_double()
                i += 1
                break

        # Length of the square we try to fit
        self.square_length = sum(self.robot_lengths)
        self.gap_finder = GapPositionFinder(scene)

    def set_bounds_manually(self, min_x, max_x, min_y, max_y):
        """
        Set the sampling bounds manually (instead of supplying a scene)
        Bounds are given in CGAL :class:`~discopygal.bindings.FT`
        """
        self.min_x, self.max_x, self.min_y, self.max_y = min_x, max_x, min_y, max_y

    def set_num_samples(self, num_samples: int):
        """
        Set the number of samples to generate
        :param num_samples:
        :type num_samples: int
        """
        pass

    def ready_sampler(self):
        """
        Prepare the sampler for sampling
        """
        pass

    def sample(self) -> Point_2:
        """
        Return a sample in the space (might be invalid)

        :return: sampled point
        :rtype: :class:`~discopygal.bindings.Point_2`
        """
        x = random.uniform(self.min_x.to_double(), self.max_x.to_double())
        y = random.uniform(self.min_y.to_double(), self.max_y.to_double())
        return Point_2(FT(x), FT(y))
