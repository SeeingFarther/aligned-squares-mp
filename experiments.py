import argparse
import json
import os
from contextlib import redirect_stdout

from discopygal.solvers import Scene
from discopygal.bindings import *
from discopygal.solvers.metrics import Metric_Euclidean as disco_metric

from utils.experiment_wrapper import ExperimentsWrapper
from metrics.max_l2_metric import Metric_Max_L2
from metrics.euclidean_metric import Metric_Euclidean
from metrics.ctd_metric import Metric_CTD
from metrics.epsilon_metric import Metric_Epsilon_2, Metric_Epsilon_Inf



class StringPrinter:
    def __init__(self):
        self.filename = None
        self.to_file = False
        return

    def ready_printer(self, args):
        self.filename = args.file
        self.to_file = args.to_file
        if self.to_file:
            self.print_func = self.print_to_file_and_stdout
            if os.path.exists(self.filename) and args.append_to_file == False:
                print('File exists')
                exit(-1)

            if args.append_to_file == False:
                with open(self.filename, 'w') as file:
                    with redirect_stdout(file):
                        print('Start of the file')
        else:
            self.print_func = self.print_to_stdout

    def print(self, input_string):
        self.print_func(input_string)

    def print_to_stdout(self, input_string):
        print(input_string)

    def print_to_file_and_stdout(self, input_string):
        # Print to stdout
        print(input_string)

        # Write to file
        with open(self.filename, 'a') as file:
            with redirect_stdout(file):
                print(input_string)


def length_k(scenes_path: list[str], solver: str, sampler: str = None, num_experiments: int = 5,
             k_values: list[int] = [5, 15, 50], nearest_neighbors_metric: str = 'CTD', roadmap_nearest_neighbors_metric: str = 'CTD',
             num_landmark: int = 1000, bound: FT = 0, delta: int = 0.1, eps: int = 9999,
             prm_num_landmarks: int = 2000, exact: bool = False, time_limit=100000):
    # Run experiments for each scene
    for scene_path in scenes_path:
        scene_name = scene_path.split('/')[-1].split('.')[0]
        with open(scene_path, 'r') as fp:
            scene = Scene.from_dict(json.load(fp))

        string_printer.print('_________________________')
        string_printer.print(f'Results for scene: {scene_name}')
        string_printer.print('For following parameters:')
        string_printer.print(f'Solver: {solver}')
        string_printer.print(f'Number of experiments: {num_experiments}')
        string_printer.print(f'num_landmark: : {num_landmark}')
        string_printer.print(f'Bounding width factor: {bound}')
        string_printer.print(f'Delta: {delta}')
        string_printer.print(f'Epsilon: {eps}')
        string_printer.print(f'PRM Number of landmarks: {prm_num_landmarks}')
        string_printer.print(f'Exact:{exact}')

        for k in k_values:
            if solver == 'PRM' or solver == 'Squares':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmark, k=k,
                                                        nearest_neighbors_metric=nearest_neighbors_metric,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        exact=exact, time_limit=time_limit)
            elif solver == 'DRRT':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmark, k=k,
                                                        nearest_neighbors_metric=nearest_neighbors_metric,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        prm_num_landmarks=prm_num_landmarks, exact=exact,
                                                        roadmap_nearest_neighbors_metric=roadmap_nearest_neighbors_metric,
                                                        time_limit=time_limit)
            elif solver == 'StaggeredGrid':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        nearest_neighbors_metric=nearest_neighbors_metric, eps=eps,
                                                        delta=delta,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        exact=exact, time_limit=time_limit)

            time, path_len, amount_of_runs = experiment_wrapper.run()
            string_printer.print(
                f'Results for k: {k}  we have got {time:.5f} seconds, {path_len} path and {amount_of_runs} runs')
    return


def length_num_landmarks(scenes_path: list[str], solver: str, sampler: str = None, num_experiments: int = 5,
                         k: int = 15, nearest_neighbors_metric: str = None,
                         roadmap_nearest_neighbors_metric: str = None,
                         num_landmarks_values: list[int] = [500, 1000, 5000], bound: FT = 0,
                         delta: int = 0.04, eps: int = 9999, prm_num_landmarks: int = 2000, exact: bool = False,
                         time_limit=100000):
    for scene_path in scenes_path:
        scene_name = scene_path.split('/')[-1].split('.')[0]
        with open(scene_path, 'r') as fp:
            scene = Scene.from_dict(json.load(fp))

        string_printer.print('_________________________')
        string_printer.print(f'Results for scene: {scene_name}')
        string_printer.print('For following parameters:')
        string_printer.print(f'Solver: {solver}')
        string_printer.print(f'Number of experiments: {num_experiments}')
        string_printer.print(f'K: : {k}')
        print(f'Roadmap nearest neighbors: {roadmap_nearest_neighbors_metric}')
        string_printer.print(f'Bounding width factor: {bound}')
        string_printer.print(f'Delta: {delta}')
        string_printer.print(f'Epsilon: {eps}')
        string_printer.print(f'PRM Number of landmarks: {prm_num_landmarks}')
        string_printer.print(f'Exact:{exact}')
        string_printer.print(f'Nearest Neighbors Metric: {nearest_neighbors_metric}')

        for num_landmarks in num_landmarks_values:
            if solver == 'PRM' or solver == 'Squares':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmarks, k=k,
                                                        nearest_neighbors_metric=nearest_neighbors_metric,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        exact=exact, time_limit=time_limit)
            elif solver == 'DRRT':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmarks, k=k,
                                                        nearest_neighbors_metric=nearest_neighbors_metric,
                                                        roadmap_nearest_neighbors_metric=roadmap_nearest_neighbors_metric,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        prm_num_landmarks=prm_num_landmarks, exact=exact,
                                                        time_limit=time_limit)
            elif solver == 'StaggeredGrid':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        nearest_neighbors_metric=nearest_neighbors_metric, eps=eps,
                                                        delta=delta,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        exact=exact, time_limit=time_limit)

            time, path_len, amount_of_runs = experiment_wrapper.run()
            string_printer.print(
                f'Results for num_landmark: {num_landmarks}  we have got {time:.5f} seconds, {path_len} path length and {amount_of_runs} runs')
    return


def length_metrics(scenes_path: list[str], solver: str, sampler: str = None, num_experiments: int = 5,
                   k: int = 15,
                   nearest_neighbors_metrics: list[str] = ['', 'Euclidean', 'CTD', 'Epsilon_2', 'Epsilon_Inf', 'Max_L2',
                                                           'Mix_CTD', 'Mix_Epsilon_2'],
                   num_landmark: int = 1000, bound: FT = 0,
                   delta: int = 0.04, eps: int = 9999, prm_num_landmarks: int = 2000, exact: bool = False,
                   time_limit=10000000):
    for scene_path in scenes_path:
        metrics_time_results = {}
        metrics_length_results = {}
        metrics_amount_of_runs_results = {}

        scene_name = scene_path.split('/')[-1].split('.')[0]
        with open(scene_path, 'r') as fp:
            scene = Scene.from_dict(json.load(fp))

        for nearest_neighbors_metric in nearest_neighbors_metrics:
            if solver == 'PRM' or solver == 'Squares':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmark, k=k,
                                                        nearest_neighbors_metric=nearest_neighbors_metric,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        exact=exact, time_limit=time_limit)
            elif solver == 'DRRT':
                roadmap_nearest_neighbors_metric = nearest_neighbors_metric
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmark, k=k,
                                                        nearest_neighbors_metric=nearest_neighbors_metric,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        prm_num_landmarks=prm_num_landmarks, exact=exact,
                                                        time_limit=time_limit,
                                                        roadmap_nearest_neighbors_metric=roadmap_nearest_neighbors_metric)
            elif solver == 'StaggeredGrid':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        nearest_neighbors_metric=nearest_neighbors_metric, eps=eps,
                                                        delta=delta,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        exact=exact, time_limit=time_limit)

            time, path_len, amount_of_runs = experiment_wrapper.run()
            metrics_length_results[nearest_neighbors_metric] = path_len
            metrics_time_results[nearest_neighbors_metric] = time
            metrics_amount_of_runs_results[nearest_neighbors_metric] = amount_of_runs

        min_time = 999999
        min_time_metric = ''
        for metric, time in metrics_time_results.items():
            if time != 0 and time < min_time:
                min_time = time
                min_time_metric = metric

        min_path_len = 999999
        min_len_metric = ''
        for metric, path_len in metrics_length_results.items():
            if path_len != 0 and path_len < min_path_len:
                min_path_len = path_len
                min_len_metric = metric

        min_amount_of_runs = 999999
        min_amount_of_runs_metric = ''
        for metric, amount_of_runs in metrics_amount_of_runs_results.items():
            if amount_of_runs != 0 and amount_of_runs < min_amount_of_runs:
                min_amount_of_runs = amount_of_runs
                min_amount_of_runs_metric = metric


        string_printer.print('_________________________')
        string_printer.print(f'Results for scene: {scene_name}')
        string_printer.print('For following parameters:')
        string_printer.print(f'Solver: {solver}')
        string_printer.print(f'Number of experiments: {num_experiments}')
        string_printer.print(f'K: : {k}')
        string_printer.print(f'Roadmap nearest neighbors: {roadmap_nearest_neighbors_metric}')
        string_printer.print(f'Number of landmarks: {num_landmark}')
        string_printer.print(f'Bounding width factor: {bound}')
        string_printer.print(f'Delta: {delta}')
        string_printer.print(f'Epsilon: {eps}')
        string_printer.print(f'PRM Number of landmarks: {prm_num_landmarks}')
        string_printer.print(f'Exact:{exact}')
        string_printer.print(f'Time Limit: {time_limit}')
        string_printer.print(f'Times:: {metrics_time_results}')
        string_printer.print(f'Path Lengths: {metrics_length_results}')
        string_printer.print(f'Amount of runs: {metrics_amount_of_runs_results}')
        string_printer.print(
            f'For {num_experiments} experiments, best time won solver {min_time_metric} with {min_time:.5f} seconds, best length won solver {min_len_metric} with {min_path_len}  path length and best amount of runs won solver {min_amount_of_runs_metric} with {min_amount_of_runs} runs')
        string_printer.print('_________________________')
    return


def compare_algo(scenes_path: list[str], solvers: list[str], sampler: str = None, num_experiments: int = 5,
                 k: int = 15, nearest_neighbors_metric: str = None, roadmap_nearest_neighbors_metric: str = None,
                 num_landmark: int = 5000, bound: FT = 0,
                 delta: int = 0.04, eps: int = 9999, prm_num_landmarks: int = 2000, exact: bool = False,
                 time_limit=100000):
    # Run experiments for each scene
    for scene_path in scenes_path:
        solvers_time_results = {}
        solvers_length_results = {}
        metrics_amount_of_runs_results = {}

        scene_name = scene_path.split('/')[-1].split('.')[0]
        with open(scene_path, 'r') as fp:
            scene = Scene.from_dict(json.load(fp))

        for solver in solvers:
            if solver == 'PRM' or solver == 'Squares':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmark, k=k,
                                                        nearest_neighbors_metric=nearest_neighbors_metric,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        exact=exact, time_limit=time_limit)
            elif solver == 'DRRT':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmark, k=k,
                                                        nearest_neighbors_metric=nearest_neighbors_metric,
                                                        roadmap_nearest_neighbors_metric=roadmap_nearest_neighbors_metric,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        prm_num_landmarks=prm_num_landmarks, exact=exact,
                                                        time_limit=time_limit)
            elif solver == 'StaggeredGrid':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        eps=eps, delta=delta,
                                                        nearest_neighbors_metric=nearest_neighbors_metric,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        exact=exact, time_limit=time_limit)

            time, path_len, runs = experiment_wrapper.run()
            solvers_length_results[solver] = path_len
            solvers_time_results[solver] = time
            metrics_amount_of_runs_results[solver] = runs

        min_time = 999999
        min_time_solver = ''
        for solver, time in solvers_time_results.items():
            if time != 0 and time < min_time:
                min_time = time
                min_time_solver = solver

        min_path_len = 999999
        min_len_solver = ''
        for solver, path_len in solvers_length_results.items():
            if path_len != 0 and path_len < min_path_len:
                min_path_len = path_len
                min_len_solver = solver

        min_amount_of_runs = 999999
        min_amount_of_runs_metric = ''
        for metric, amount_of_runs in metrics_amount_of_runs_results.items():
            if amount_of_runs != 0 and amount_of_runs < min_amount_of_runs:
                min_amount_of_runs = amount_of_runs
                min_amount_of_runs_metric = metric

        string_printer.print('_________________________')
        string_printer.print(f'Results for scene: {scene_name}')
        string_printer.print('For following parameters:')
        string_printer.print(f'Solver: {solvers}')
        string_printer.print(f'Number of experiments: {num_experiments}')
        string_printer.print(f'K: : {k}')
        string_printer.print(f'Roadmap nearest neighbors: {roadmap_nearest_neighbors_metric}')
        string_printer.print(f'Number of landmarks: {num_landmark}')
        string_printer.print(f'Bounding width factor: {bound}')
        string_printer.print(f'Delta: {delta}')
        string_printer.print(f'Epsilon: {eps}')
        string_printer.print(f'PRM Number of landmarks: {prm_num_landmarks}')
        string_printer.print(f'Exact:{exact}')
        string_printer.print(f'Time Limit: {time_limit}')
        string_printer.print(f'Times:: {solvers_time_results}')
        string_printer.print(f'Path Lengths: {solvers_length_results}')
        string_printer.print(f'Nearest Neighbors Metric: {nearest_neighbors_metric}')
        string_printer.print(f'Roadmap Nearest Neighbors Metric: {roadmap_nearest_neighbors_metric}')
        string_printer.print(
            f'For {num_experiments} experiments, best time won solver {min_time_solver} with {min_time:.5f} seconds, best length won solver {min_len_solver} with {min_path_len}  path length and best amount of runs won solver {min_amount_of_runs_metric} with {min_amount_of_runs} runs')
        string_printer.print('_________________________')
    return


def get_scene_paths(scene_dir: str):
    scenes = []
    for root, dirs, files in os.walk(scene_dir, topdown=False):
        for file in files:
            if file.endswith('.json'):
                scenes.append(os.path.join(root, file))
    return scenes


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser")

    # Add arguments
    parser.add_argument('--compare-landmarks', action='store_true', default=False, help='Do landmarks hyperparameter experiments')
    parser.add_argument('--compare-algo', action='store_true', default=False, help='Do algorithms experiments')
    parser.add_argument('--compare-length', action='store_true', default=False, help='Do metric length experiments')
    parser.add_argument('--compare-k', action='store_true', default=False, help='Do k hyperparameter experiments')
    parser.add_argument('--k', type=int, default=15, help='Value of k')
    parser.add_argument('--num-landmarks', type=int, default=1000, help='Number of landmarks')
    parser.add_argument('-prm-num-landmarks', type=int, default=2000, help='Number of landmarks for PRM for DRRT')
    parser.add_argument('--num_experiments', type=int, default=5, help='Number of experiments')
    parser.add_argument('--bound', type=int, default=2, help='Bounding width factor')
    parser.add_argument('--eps', type=float, default=5, help='Number of experiments')
    parser.add_argument('--delta', type=float, default=2, help='Bounding width factor')
    parser.add_argument('--solver', type=str, default="squares", choices=['prm', 'drrt', 'staggered', 'squares'],
                        help='Type of solver')
    parser.add_argument('--nearest_neighbors', type=str, default=None,
                        choices=['CTD', 'Euclidean', 'Epsilon_2', 'Epsilon_Inf', 'Max_L2', 'Mix_CTD', 'Mix_Epsilon_2'],
                        help='Type of solver')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['CTD', 'Euclidean', 'Epsilon_2', 'Epsilon_Inf', 'Max_L2'],
                        help='Type of metric for edge length')
    parser.add_argument('--roadmap_nearest_neighbors', type=str, default=None,
                        choices=['CTD', 'Euclidean', 'Epsilon_2', 'Epsilon_Inf', 'Max_L2', 'Mix_CTD', 'Mix_Epsilon_2'],
                        help='Type of solver')
    parser.add_argument('--exact',  action='store_true', help='Run exact number of successful experiments')
    parser.add_argument('--path', type=str, default='./scenes/Easy2.json', help='Path to scene file')
    parser.add_argument('--sampler', type=str, default=None, choices=['uniform', 'combined'],
                        help='Type of sampler')
    parser.add_argument('--to-file', action='store_true', help='Write output to file')
    parser.add_argument('--file', type=str, default='./results/other_benchmarks_tests.txt', help='Path to scene file, not valid if with compare flags')
    parser.add_argument('--append-to-file', action='store_true', help='If file exist to append the output to him')
    parser.add_argument('--scene-dir', type=str, default='./scenes/', help='Path to scene directory for experiments sections used with compare flags')
    parser.add_argument('--time_limit', type=int, default=200, help='Time limit(seconnds) for each experiment')

    # Parse arguments
    args = parser.parse_args()
    return args


def start_running(args):

    if args.metric is None or args.metric == '':
        args.metric = disco_metric()
    elif args.metric == 'Euclidean':
        args.metric = Metric_Euclidean()
    elif args.metricc == 'CTD':
        args.metric = Metric_CTD()
    elif args.metric == 'Epsilon_2':
        args.metric = Metric_Epsilon_2()
    elif args.metric == 'Epsilon_Inf':
        args.metric = Metric_Epsilon_Inf()
    elif args.metric== 'Max_L2':
        args.metric = Metric_Max_L2()

    if args.solver == 'drrt':
        solver = 'DRRT'
    elif args.solver == 'staggered':
        solver = 'StaggeredGrid'
    elif args.solver == 'squares':
        solver = 'Squares'
    elif args.solver == 'prm':
        solver = 'PRM'
    else:
        print('No such solver')
        exit(-1)

    if args.compare_algo:
        scenes = get_scene_paths(args.scene_dir)
        compare_algo(scenes, ['PRM', 'DRRT', 'StaggeredGrid', 'Squares'], None, num_experiments=args.num_experiments, k=args.k, num_landmark=args.num_landmarks, bound=args.bound, delta=args.delta, eps=args.eps, prm_num_landmarks=args.prm_num_landmarks)
        exit(0)

    if args.compare_length:
        scenes = get_scene_paths(args.scene_dir)
        if args.solver == "drrt":
            length_metrics(scenes, solver=solver, num_experiments=args.num_experiments, bound=args.bound,
                                 k=args.k, nearest_neighbors_metrics=['', 'Euclidean'], delta=args.delta,
                                 eps=args.eps,
                                 prm_num_landmarks=args.prm_num_landmarks, exact=args.exact,
                                 time_limit=args.time_limit)
        elif args.solver == "staggered":
            print('No nearest neighbors metric for Staggered Grid')
        else:
            length_metrics(scenes, solver=solver, num_experiments=args.num_experiments, bound=args.bound,
                                 k=args.k, delta=args.delta,
                                 eps=args.eps,
                                 prm_num_landmarks=args.prm_num_landmarks, exact=args.exact,
                                 time_limit=args.time_limit)
        exit(0)

    if args.compare_landmarks:
        scenes = get_scene_paths(args.scene_dir)
        length_num_landmarks(scenes, solver=solver, num_experiments=args.num_experiments, bound=args.bound,
                             k=args.k, nearest_neighbors_metric=args.nearest_neighbors,
                             roadmap_nearest_neighbors_metric=args.roadmap_nearest_neighbors, delta=args.delta,
                             eps=args.eps,
                             prm_num_landmarks=args.prm_num_landmarks, exact=args.exact,
                             time_limit=args.time_limit)
        exit(0)

    if args.compare_k:
        scenes = get_scene_paths(args.scene_dir)
        length_k(scenes, solver=solver, num_experiments=args.num_experiments, bound=args.bound,
                 num_landmark=args.num_landmarks, nearest_neighbors_metric=args.nearest_neighbors,
                 roadmap_nearest_neighbors_metric=args.roadmap_nearest_neighbors, delta=args.delta,
                 eps=args.eps,
                 prm_num_landmarks=args.prm_num_landmarks, exact=args.exact, time_limit=args.time_limit)
        exit(0)

    experiment_wrapper = None
    with open(args.path, 'r') as fp:
        scene = Scene.from_dict(json.load(fp))
    if args.solver == 'prm':
        experiment_wrapper = ExperimentsWrapper(scene, 'PRM', num_experiments=args.num_experiments,
                                                num_landmarks=args.num_landmarks, k=args.k,
                                                nearest_neighbors_metric=args.nearest_neighbors,
                                                bounding_margin_width_factor=args.bound, sampler=args.sampler,
                                                metric=args.metric)
    elif args.solver == 'drrt':
        experiment_wrapper = ExperimentsWrapper(scene, 'DRRT', num_experiments=args.num_experiments,
                                                prm_num_landmarks=args.prm_num_landmarks,
                                                num_landmarks=args.num_landmarks, k=args.k,
                                                roadmap_nearest_neighbors_metric=args.roadmap_nearest_neighbors,
                                                nearest_neighbors_metric=args.nearest_neighbors,
                                                bounding_margin_width_factor=args.bound, sampler=args.sampler,
                                                metric=args.metric)
    elif args.solver == 'staggered':
        experiment_wrapper = ExperimentsWrapper(scene, 'StaggeredGrid', num_experiments=args.num_experiments,
                                                eps=args.eps, delta=args.delta, bounding_margin_width_factor=args.bound,
                                                sampler=args.sampler, metric=args.metric)
    elif args.solver == 'squares':
        experiment_wrapper = ExperimentsWrapper(scene, 'Squares', num_experiments=args.num_experiments,
                                                num_landmarks=args.num_landmarks, k=args.k,
                                                nearest_neighbors_metric=args.nearest_neighbors,
                                                bounding_margin_width_factor=args.bound, sampler=args.sampler,
                                                metric=args.metric)

    time_result, path_len_result, total_runs = experiment_wrapper.run()
    string_printer.print(
        f'Results for {args.num_experiments} experiments, for solver {args.solver} we have got {time_result:.3f} seconds and {path_len_result} path length, we run {total_runs} times')


string_printer = StringPrinter()

if __name__ == '__main__':
    args = parse_arguments()
    string_printer.ready_printer(args)
    start_running(args)
