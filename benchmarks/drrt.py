from discopygal.solvers import Scene
from discopygal.bindings import *
from discopygal.solvers.rrt.drrt import dRRT


class BasicDRRTForExperiments(dRRT):
    def load_scene(self, scene: Scene):
        super().load_scene(scene)

    def solve(self):
        path = super().solve()
        return path