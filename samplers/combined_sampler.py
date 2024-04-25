
from samplers.basic_sampler import BasicSquaresSampler
from samplers.pair_sampler import PairSampler
from samplers.space_sampler import SpaceSampler


class CombinedSampler(BasicSquaresSampler):

    def __init__(self, scene=None, num_landmarks: int = None, percentage: float = 0.2):
        super().__init__(scene)
        self.percentage = percentage
        self.num_landmarks = num_landmarks
        self.limit = int(num_landmarks * self.percentage) if num_landmarks is not None else None
        self.num_sample = 0
        self.scene = scene
        self.first_sampler = PairSampler(scene)
        self.second_sampler = SpaceSampler(scene)

    def set_scene(self, scene,  bounding_box = None):
        super().set_scene(scene, bounding_box)
        self.scene = scene
        self.first_sampler.set_scene(scene)
        self.second_sampler.set_scene(scene)

    def sample_free(self):
        self.num_sample += 1
        if self.num_sample < self.limit:
            return self.first_sampler.sample_free()
        return self.second_sampler.sample_free()
    
    def ready_sampler(self):
        super().ready_sampler()
        self.first_sampler.ready_sampler()
        self.second_sampler.ready_sampler()

    def set_num_samples(self, num_landmarks: int):
        self.num_landmarks = num_landmarks
        self.limit = int(num_landmarks * self.percentage)
        self.first_sampler.set_num_samples(self.limit)
        self.second_sampler.set_num_samples(num_landmarks - self.limit)
