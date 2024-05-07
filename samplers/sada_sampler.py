import json

import numpy as np
from discopygal.bindings import Point_2
from discopygal.solvers import Scene

from samplers.basic_sampler import BasicSquaresSampler
from samplers.bridge_sampler import BridgeSampler
from samplers.middle_sampler import MiddleSampler
from samplers.space_sampler import SpaceSampler


class SadaSampler(BasicSquaresSampler):
    def __init__(self, samplers: list[BasicSquaresSampler], scene: Scene = None, gamma: float = 0.2):
        """
        Constructor for the SadaSampler.
        :param samplers:
        :type samplers: list
        :param scene:
        :type scene: :class:`~discopygal.solvers.Scene`
        :param fixed_constant:
        :type fixed_constant: float
        """
        # Initialize the sampler
        super().__init__(scene)
        self.scene = scene
        self.num_sample = 0
        self.sampler_index = 0
        self.num_samplers = len(samplers)
        self.samplers = samplers
        self.gamma = gamma
        self.K = 1. / self.num_samplers,
        self.weights = [np.ones(self.num_samplers), np.ones(self.num_samplers)]
        self.cost_weights = [np.ones(self.num_samplers), np.ones(self.num_samplers)]
        self.pi_sensitive = [np.ones(self.num_samplers), np.ones(self.num_samplers)]

        self.sampler_index = 0

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

    def compute_cost_insensitive_probability(self, robot_index: int) -> np.array:
        """
        Compute the cost insensitive probability for the sampler.
        :param robot_index:
        :type robot_index: int

        :return: Cost insensitive probability
        :rtype: np.array
        """
        weights = self.weights[robot_index]
        total_weight = np.sum(weights)
        return np.array([(1 - self.gamma) * (weights[i] / total_weight) + (self.gamma * (1 / self.num_samplers))
                         for i in range(self.num_samplers)])

    def update_weights(self, robot_index: int, reward: float) -> None:
        """
        Update the weights of the sampler based on the reward.
        :param robot_index:
        :type robot_index: int
        :param reward:
        :type reward: float

        :return: None
        :rtype: None
        """
        adjusted_reward = reward / self.pi_sensitive[robot_index][self.sampler_index]
        self.weights[robot_index][self.sampler_index] *= np.exp((self.gamma * adjusted_reward) / self.num_samplers)

    def compute_cost_sensitive_probability(self, robot_index: int) -> None:
        """
        Compute the cost sensitive probability for the sampler.
        :param robot_index:
        :type robot_index: int

        :return: None
        :rtype: None
        """
        pi = self.compute_cost_insensitive_probability(robot_index)
        cost_weights = self.cost_weights[robot_index]
        total_pi_cost = np.sum(pi / cost_weights)
        result = np.array([(pi[i] / total_pi_cost) for i in range(self.num_samplers)])
        for i in range(self.num_samplers):
            self.pi_sensitive[robot_index][i] = result[i]

    def sample_free(self, robot_index: int) -> Point_2:
        """
        Sample a free point in the scene using the SadaSampler.
        Choose a sampler based on the probabilities and sample a point from it.
        :param robot_index:
        :type robot_index: int

        :return: Sampled point
        :rtype: :class:`~discopygal.bindings.Point_2`
        """
        self.compute_cost_sensitive_probability(robot_index)
        self.sampler_index = np.random.choice(range(self.num_samplers), p=self.pi_sensitive[robot_index])
        return self.samplers[self.sampler_index].sample_free(robot_index)

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

    def ready_sampler(self):
        """
        Ready the sampler for sampling.
        """
        super().ready_sampler()
        for sampler in self.samplers:
            sampler.ready_sampler()