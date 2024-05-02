import random
import numpy as np

from discopygal.solvers import Scene

from samplers.basic_sampler import BasicSquaresSampler


class CombinedSampler(BasicSquaresSampler):
    """
    Combined sampler for sampling points in a scene.
    """

    def __init__(self, probs: list[int], samplers: list[BasicSquaresSampler], scene: Scene = None):
        """
        Constructor for the CombinedSampler.
        :param probs:
        :type probs: list
        :param samplers:
        :type samplers: list
        :param scene:
        :type scene: :class:`~discopygal.solvers.Scene`
        """

        # Initialize the sampler
        super().__init__(scene)
        self.scene = scene
        self.num_sample = 0
        self.sampler_index = 0
        self.samplers = samplers
        self.init_probs = probs

    def set_scene(self, scene: Scene, bounding_box=None):
        """
        Set the scene for sampler should use for each of is samplers.
        Can be overridded to add additional processing.

        :param bounding_box:
        :type :class:`~discopygal.Bounding_Box
        :param scene: a scene to sample in
        :type scene: :class:`~discopygal.solvers.Scene`
        """

        super().set_scene(scene, bounding_box)
        self.scene = scene
        for sampler in self.samplers:
            sampler.set_scene(scene, bounding_box)

        self.probs = [self.init_probs] * len(self.robots)

    def sample_free(self, robot_index: int):
        """
        Sample a free point in the scene using the combined sampler.
        Choose a sampler based on the probabilities and sample a point from it.
        :param robot_index:
        :type robot_index: int

        :return: Sampled point
        :rtype: :class:`~discopygal.bindings.Point_2`
        """

        # Choose a sampler based on the probabilities
        indexes = list(range(len(self.samplers)))
        self.sampler_index = random.choices(indexes, self.probs[robot_index], k=1)[0]

        # Sample a point from the chosen sampler
        return self.samplers[self.sampler_index].sample_free(robot_index)

    def set_init_probs(self, probs: list[int]):
        """
        Set the probabilities for each sampler.
        :param probs:
        :type probs: list
        """
        self.init_probs = probs

    def ready_sampler(self):
        """
        Ready the sampler for sampling.
        """
        super().ready_sampler()
        for sampler in self.samplers:
            sampler.ready_sampler()

    def set_num_samples(self, num_samples: int):
        """
        Set the number of samples to generate.
        :param num_samples:
        :type num_samples: int
        """
        super().set_num_samples(num_samples)
        for sampler in self.samplers:
            sampler.set_num_samples(num_samples)
        return

    def update_probs(self, robot_index: int, reward: float, learning_rate: float = 0.001):
        """
        Update the probabilities of the samplers based on the reward.
        :param robot_index:
        :type robot_index: int
        :param reward:
        :type reward: float
        :param learning_rate:
        :type learning_rate: float
        :return:
        """
        # Update probabilities using reward-based learning
        self.probs[robot_index][self.sampler_index] += learning_rate * reward
        self.probs[robot_index] /= np.sum(self.probs[robot_index])  # Normalize probabilities to sum to 1
