from discopygal.solvers import Scene, PathCollection
from discopygal.solvers.prm import PRM


class BasicPrmForExperiments(PRM):
    """
    Basic PRM implementation for experiments
    """
    def load_scene(self, scene: Scene):
        """
        Load a scene for the PRM algorithm
        :param scene: :class:`~discopygal.solvers.Scene`
        """
        super().load_scene(scene)

    def solve(self) -> PathCollection:
        """
        Solve the scene using the PRM algorithm
        :return: Path solution
        :rtype: :class:`~discopygal.solvers.PathCollection`
        """

        path = super().solve()
        return path
