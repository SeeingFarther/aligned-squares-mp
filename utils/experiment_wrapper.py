import time

from discopygal.solvers import Scene, PathCollection
from discopygal.solvers.metrics import Metric, Metric_Euclidean
from discopygal.solvers.Solver import Solver
from discopygal.solvers.samplers import Sampler
from discopygal.bindings import FT, Point_2

from benchmarks.drrt import BasicDRRTForExperiments
from benchmarks.prm import BasicPrmForExperiments
from benchmarks.staggered_grid import BasicsStaggeredGridForExperiments
from squares_planner import SquaresPrm as SquareMotionPlanner


class ExperimentsWrapper:
    """
    Wrapper for running experiments
    """

    def __init__(self, scene: Scene, solver_name: str, num_experiments: int = 5, num_landmarks: int = -1,
                 k: int = -1,
                 eps: float = -1, delta: float = -1,
                 bounding_margin_width_factor: FT = Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
                 nearest_neighbors=None, metric: Metric = None, sampler: Sampler = None, prm_num_landmarks=None,
                 prm_nearest_neighbors=None):
        """
        Constructor for the ExperimentsWrapper.

        :param scene:
        :type scene: :class:`~discopygal.solvers.Scene`
        :param solver_name
        :type solver_name: str
        :param num_experiments:
        :type num_experiments: int
        :param num_landmarks:
        :type num_landmarks: int
        :param k:
        :type k: int
        :param eps:
        :type eps: float
        :param delta:
        :type delta: float
        :param bounding_margin_width_factor:
        :type bounding_margin_width_factor: :class:`~discopygal.bindings.FT`
        :param nearest_neighbors:
        :type nearest_neighbors:
        :param metric:
        :type metric: :class:`~discopygal.solvers.metrics.Metric`
        :param sampler:
        :type sampler: :class:`~discopygal.solvers.samplers.Sampler`
        :param prm_num_landmarks:
        :type prm_num_landmarks: int
        :param prm_nearest_neighbors:
        :type prm_nearest_neighbors:
        """
        # Build the proper solver
        if solver_name == 'PRM':
            self.solver = BasicPrmForExperiments(num_landmarks, k,
                                                 bounding_margin_width_factor=bounding_margin_width_factor,
                                                 nearest_neighbors=nearest_neighbors, metric=metric, sampler=sampler)
        elif solver_name == 'DRRT':
            self.solver = BasicDRRTForExperiments(num_landmarks=num_landmarks, prm_num_landmarks=prm_num_landmarks,
                                                  prm_k=k,
                                                  bounding_margin_width_factor=bounding_margin_width_factor,
                                                  nearest_neighbors=nearest_neighbors,
                                                  prm_nearest_neighbors=prm_nearest_neighbors, metric=metric,
                                                  sampler=sampler)
        elif solver_name == 'StaggeredGrid':
            self.solver = BasicsStaggeredGridForExperiments(eps, delta,
                                                            bounding_margin_width_factor=bounding_margin_width_factor,
                                                            nearest_neighbors=nearest_neighbors, metric=metric,
                                                            sampler=sampler)
        elif solver_name == 'Squares':
            self.solver = SquareMotionPlanner(num_landmarks=num_landmarks, k=k,
                                              bounding_margin_width_factor=Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
                                              sampler=sampler)
        else:
            raise ValueError('Invalid solver name')

        # Load scene
        self.solver.load_scene(scene)
        self.num_experiments = num_experiments

        # Metric for path length
        self.metric = metric
        if self.metric is None:
            self.metric = Metric_Euclidean

    def run(self) -> (float, float):
        """
        Run the experiments

        :return: Average time and path length
        :rtype float, float
        """
        time_results = []
        paths_len = []

        # Run the experiments
        for experiment in range(self.num_experiments):
            # Compute time of execution
            start = time.time()
            robot_paths = self.solver.solve()
            total_time = time.time() - start
            time_results.append(total_time)

            # Compute path length
            paths_len.append(path_length(robot_paths, self.metric.dist))

        # Compute average time and path length
        avg_time = avg(time_results)
        avg_path = avg(paths_len)
        return avg_time, avg_path


def avg(lst: list) -> float:
    """
    Compute the average of a list of numbers
    :param lst:
    :type lst: list

    :return: average of the numbers in the list
    :rtype float
    """

    # Empty list? return 0
    if not lst:
        return 0

    return sum(lst) / len(lst)


def path_length(robot_paths: PathCollection, metric: Metric.dist) -> float:
    """
    Compute the length of a path
    :param robot_paths:
    :type robot_paths: PathCollection
    :param metric:
    :type metric: Metric.dist

    :return: length of the path
    :rtype float
    """
    w = 0

    # Iterate over the paths nodes and add them as part of the path
    paths = robot_paths.paths
    for key, path in paths.items():
        nodes = path.points

        # Compute length of the path
        for ind, nd in enumerate(nodes[1:]):
            prev = nodes[ind].location
            current = nd.location

            point = Point_2(FT(current[0]), FT(current[1]))
            neighbor = Point_2(FT(prev[0]), FT(prev[1]))

            # Compute distance
            dist = metric(point, neighbor)

            if isinstance(dist, FT):
                w += dist.to_double()
            else:
                w += dist
    return w
