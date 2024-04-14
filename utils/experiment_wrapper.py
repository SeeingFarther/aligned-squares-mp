import time

from discopygal.solvers import Scene
from discopygal.solvers.metrics import Metric, Metric_Euclidean
from discopygal.solvers.Solver import Solver
from discopygal.solvers.samplers import Sampler
from discopygal.bindings import FT, Point_2

from benchmarks.drrt import BasicDRRTForExperiments
from benchmarks.prm import BasicPrmForExperiments
from benchmarks.staggered_grid import BasicsStaggeredGridForExperiments
from squares_planner import SquareMotionPlanner


class ExperimentsWrapper:
    def __init__(self, scene: Scene, solver: str, num_experiments: int = 5, num_landmarks: int = -1,
                 k: int = -1,
                 eps: float = -1, delta: float = -1,
                 bounding_margin_width_factor: FT = Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
                 nearest_neighbors=None, metric: Metric = None, sampler: Sampler = None, prm_num_landmarks=None,
                 prm_nearest_neighbors=None):

        if solver == 'PRM':
            self.solver = BasicPrmForExperiments(num_landmarks, k,
                                                 bounding_margin_width_factor=bounding_margin_width_factor,
                                                 nearest_neighbors=nearest_neighbors, metric=metric, sampler=sampler)
        elif solver == 'DRRT':
            self.solver = BasicDRRTForExperiments(num_landmarks=num_landmarks, prm_num_landmarks=prm_num_landmarks,
                                                  prm_k=k,
                                                  bounding_margin_width_factor=bounding_margin_width_factor,
                                                  nearest_neighbors=nearest_neighbors,
                                                  prm_nearest_neighbors=prm_nearest_neighbors, metric=metric,
                                                  sampler=sampler)
        elif solver == 'StaggeredGrid':
            self.solver = BasicsStaggeredGridForExperiments(eps, delta,
                                                            bounding_margin_width_factor=bounding_margin_width_factor,
                                                            nearest_neighbors=nearest_neighbors, metric=metric,
                                                            sampler=sampler)
        elif solver == 'Squares':
            self.solver = SquareMotionPlanner(num_landmarks=num_landmarks, k=k,
                                              bounding_margin_width_factor=Solver.DEFAULT_BOUNDS_MARGIN_FACTOR,
                                              sampler=sampler)
        else:
            raise ValueError('Invalid solver name')

        self.solver.load_scene(scene)
        self.num_experiments = num_experiments

        self.metric = metric
        if self.metric is None:
            self.metric = Metric_Euclidean

    def run(self) -> (float, float):
        time_results = []
        paths_len = []
        for experiment in range(self.num_experiments):
            start = time.time()
            robot_paths = self.solver.solve()
            total_time = time.time() - start
            time_results.append(total_time)
            paths_len.append(path_length(robot_paths, self.metric.dist))

        avg_time = avg(time_results)
        avg_path = avg(paths_len)
        return avg_time, avg_path


def avg(lst: list):
    if not lst:
        return 0
    return sum(lst) / len(lst)


def path_length(robot_paths, metric):
    w = 0
    paths = robot_paths.paths
    for key, path in paths.items():
        nodes = path.points

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
