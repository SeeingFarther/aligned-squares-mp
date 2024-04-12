from discopygal.solvers import Scene
from discopygal.solvers.prm import PRM


class BasicPrmForExperiments(PRM):
    def load_scene(self, scene: Scene):
        super().load_scene(scene)

    def solve(self):
        path = super().solve()
        return path
