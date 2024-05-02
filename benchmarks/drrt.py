from discopygal.solvers import Scene, PathCollection
from discopygal.solvers.rrt.drrt import dRRT


class BasicDRRTForExperiments(dRRT):
    """ Basic DRRT for experiments"""

    def load_scene(self, scene: Scene):
        """
        Load the scene for the DRRT algorithm
        :param scene
        :type scene: :class:`~discopygal.solvers.Scene`
        """
        super().load_scene(scene)

    def solve(self) -> PathCollection:
        """
        Solve the scene using the DRRT algorithm
        :return: Path solution
        :rtype: :class:`~discopygal.solvers.PathCollection`
        """
        path = super().solve()
        return path
