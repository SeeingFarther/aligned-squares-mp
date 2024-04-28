import random

import numpy as np

from samplers.basic_sampler import BasicSquaresSampler
from samplers.pair_sampler import PairSampler
from samplers.space_sampler import SpaceSampler


class CombinedSampler(BasicSquaresSampler):

    def __init__(self, probs: list, samplers: list, scene=None):
        super().__init__(scene)
        self.scene = scene
        self.num_sample = 0
        self.sampler_index = 0
        self.samplers = samplers
        self.probs = probs

    def set_scene(self, scene, bounding_box=None):
        super().set_scene(scene, bounding_box)
        self.scene = scene
        for sampler in self.samplers:
            sampler.set_scene(scene, bounding_box)

    def sample_free(self, robot_index):
        indexes = list(range(len(self.samplers)))
        self.sampler_index = random.choices(indexes, self.probs, k=1)[0]
        return self.samplers[self.sampler_index].sample_free(robot_index)

    def set_probs(self, probs):
        self.probs = probs

    def ready_sampler(self):
        super().ready_sampler()
        for sampler in self.samplers:
            sampler.ready_sampler()

    def set_num_samples(self, num_samples: int):
        super().set_num_samples(num_samples)
        for sampler in self.samplers:
            sampler.set_num_samples(num_samples)
        return

    def update_probs(self, sampler_index, reward, learning_rate=0.1):
        # Update probabilities using reward-based learning
        self.probs[sampler_index] += learning_rate * reward
        self.probs /= np.sum(self.probs)  # Normalize probabilities to sum to 1
