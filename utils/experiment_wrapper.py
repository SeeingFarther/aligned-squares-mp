import time

import numpy as np
from discopygal.solvers import Scene, PathCollection
from discopygal.solvers.metrics import Metric, Metric_Euclidean
from discopygal.solvers.Solver import Solver
from discopygal.solvers.nearest_neighbors import NearestNeighbors_sklearn
from discopygal.solvers.samplers import Sampler
from discopygal.bindings import FT, Point_2

from benchmarks.drrt import BasicDRRTForExperiments
from benchmarks.prm import BasicPrmForExperiments
from benchmarks.staggered_grid import BasicsStaggeredGridForExperiments
from metrics.ctd_metric import Metric_CTD
from metrics.epsilon_metric import Metric_Epsilon_Inf, Metric_Epsilon_2
from squares_planner import SquaresPrm as SquareMotionPlanner
from utils.nearest_neighbors import NearestNeighbors_sklearn_ball


class ExperimentsWrapper:
    """
    Wrapper for running experiments
    """

    def __init__(self, scene: Scene, solver_name: str, num_experiments: int = 5, num_landmarks: int = -1,
                 k: int = -1,
                 eps: float = -1, delta: float = -1,
                 bounding_margin_width_factor: FT = Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
                 nearest_neighbors_metric: str = 'CTD',
                 metric: Metric = None, sampler: Sampler = None, prm_num_landmarks=None,
                 exact: bool = False, wrapper_metric: Metric = None, time_limit: float = 100000):
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
        :param nearest_neighbors_metric:
        :type nearest_neighbors_metric: str
        :param metric:
        :type metric: :class:`~discopygal.solvers.metrics.Metric`
        :param sampler:
        :type sampler: :class:`~discopygal.solvers.samplers.Sampler`
        :param prm_num_landmarks:
        :type prm_num_landmarks: int

        """
        # Build the proper solver
        self.solver_name = solver_name
        self.bounding_margin_width_factor = bounding_margin_width_factor
        self.sampler = sampler
        self.prm_k = k
        self.eps = eps
        self.delta = delta
        self.k = k
        self.num_landmarks = num_landmarks
        self.prm_num_landmarks = prm_num_landmarks
        self.metric = metric
        self.time_limit = time_limit
        self.nearest_neighbors_metric = nearest_neighbors_metric

        if self.nearest_neighbors_metric is None:
            self.nearest_neighbors = NearestNeighbors_sklearn()
            self.prm_nearest_neighbors = NearestNeighbors_sklearn()
        elif nearest_neighbors_metric == 'Euclidean':
            self.nearest_neighbors = NearestNeighbors_sklearn_ball(Metric_Euclidean)
            self.prm_nearest_neighbors = NearestNeighbors_sklearn_ball(Metric_Euclidean)
        elif nearest_neighbors_metric == 'CTD':
            self.nearest_neighbors = NearestNeighbors_sklearn_ball(Metric_CTD)
            self.prm_nearest_neighbors = NearestNeighbors_sklearn_ball(Metric_CTD)
        elif nearest_neighbors_metric == 'Epsilon_2':
            self.nearest_neighbors = NearestNeighbors_sklearn_ball(Metric_Epsilon_2)
            self.prm_nearest_neighbors = NearestNeighbors_sklearn_ball(Metric_Epsilon_2)
        elif nearest_neighbors_metric == 'Epsilon_Inf':
            self.nearest_neighbors = NearestNeighbors_sklearn_ball(Metric_Epsilon_Inf)
            self.prm_nearest_neighbors = NearestNeighbors_sklearn_ball(Metric_Epsilon_Inf)
        else:
            print('Unknown metric')
            exit(-1)

        if solver_name == 'PRM':
            self.solver = BasicPrmForExperiments(num_landmarks, k,
                                                 bounding_margin_width_factor=bounding_margin_width_factor,
                                                 nearest_neighbors=self.nearest_neighbors, metric=metric,
                                                 sampler=sampler)
        elif solver_name == 'DRRT':
            self.solver = BasicDRRTForExperiments(num_landmarks=num_landmarks, prm_num_landmarks=prm_num_landmarks,
                                                  prm_k=k,
                                                  bounding_margin_width_factor=bounding_margin_width_factor,
                                                  nearest_neighbors=self.nearest_neighbors,
                                                  prm_nearest_neighbors=self.prm_nearest_neighbors, metric=metric,
                                                  sampler=sampler)
        elif solver_name == 'StaggeredGrid':
            self.solver = BasicsStaggeredGridForExperiments(eps, delta,
                                                            bounding_margin_width_factor=bounding_margin_width_factor,
                                                            nearest_neighbors=self.nearest_neighbors, metric=metric,
                                                            sampler=sampler)
        elif solver_name == 'Squares':
            self.solver = SquareMotionPlanner(num_landmarks=num_landmarks, k=k,
                                              bounding_margin_width_factor=bounding_margin_width_factor,
                                              sampler=sampler)
        else:
            raise ValueError('Invalid solver name')

        self.scene = scene
        self.num_experiments = num_experiments
        self.exact = exact
        if self.exact:
            print(
                'Exact mode enabled, until the exact number of successful experiments will not stop or time limit passed.')

        # Metric for path length
        self.wrapper_metric = wrapper_metric
        if self.wrapper_metric is None:
            self.wrapper_metric = Metric_Euclidean

    def restart(self):
        """
        Restart the solver
        """
        # Build the proper solver
        if self.solver_name == 'PRM':
            self.solver = BasicPrmForExperiments(self.num_landmarks, self.k,
                                                 bounding_margin_width_factor=self.bounding_margin_width_factor,
                                                 nearest_neighbors=self.nearest_neighbors, metric=self.metric,
                                                 sampler=self.sampler)
        elif self.solver_name == 'DRRT':
            self.solver = BasicDRRTForExperiments(num_landmarks=self.num_landmarks,
                                                  prm_num_landmarks=self.prm_num_landmarks,
                                                  prm_k=self.k,
                                                  bounding_margin_width_factor=self.bounding_margin_width_factor,
                                                  nearest_neighbors=self.nearest_neighbors,
                                                  prm_nearest_neighbors=self.prm_nearest_neighbors, metric=self.metric,
                                                  sampler=self.sampler)
        elif self.solver_name == 'StaggeredGrid':
            self.solver = BasicsStaggeredGridForExperiments(self.eps, self.delta,
                                                            bounding_margin_width_factor=self.bounding_margin_width_factor,
                                                            nearest_neighbors=self.nearest_neighbors,
                                                            metric=self.metric,
                                                            sampler=self.sampler)
        elif self.solver_name == 'Squares':
            self.solver = SquareMotionPlanner(num_landmarks=self.num_landmarks, k=self.k,
                                              bounding_margin_width_factor=self.bounding_margin_width_factor,
                                              sampler=self.sampler)

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
            stopper = time.time()
            first_time = True
            continue_running = False
            total_time = np.inf
            length = np.inf
            while continue_running or first_time:
                first_time = False
                self.restart()
                # Compute time of execution
                start = time.time()
                self.solver.load_scene(self.scene)
                robot_paths = self.solver.solve()
                total_time = time.time() - start

                # Compute path length
                length = path_length(robot_paths, self.wrapper_metric.dist)
                continue_running = self.exact and (length == 0) and (abs(time.time() - stopper) < self.time_limit)

            time_results.append(total_time)
            paths_len.append(length)

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
