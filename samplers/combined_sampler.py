from samplers.basic_sampler import BasicSquaresSampler
from samplers.pair_sampler import PairSampler
from samplers.space_sampler import SpaceSampler


class CombinedSampler(BasicSquaresSampler):

    def __init__(self, sampler_sample_num: list, samplers: list, scene=None):
        super().__init__(scene)
        self.scene = scene
        self.num_sample = 0
        self.sampler_index = 0
        self.samplers = samplers
        self.sampler_sample_num = sampler_sample_num

    def set_scene(self, scene, bounding_box=None):
        super().set_scene(scene, bounding_box)
        self.scene = scene
        for sampler in self.samplers:
            sampler.set_scene(scene, bounding_box)

    def sample_free(self):
        self.num_sample += 1
        if self.num_sample > self.sampler_sample_num[self.sampler_index]:
            self.num_sample = 0
            self.sampler_index += 1
        return self.samplers[self.sampler_index].sample_free()

    def ready_sampler(self):
        super().ready_sampler()
        for sampler in self.samplers:
            sampler.ready_sampler()

    def set_num_samples(self, num_samples: int):
        super().set_num_samples(num_samples)
        for index, sampler in enumerate(self.samplers):
            sampler.set_num_samples(self.sampler_sample_num[index])