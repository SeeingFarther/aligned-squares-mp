import argparse
import json
import os
from contextlib import redirect_stdout

from discopygal.solvers import Scene
from discopygal.bindings import *
from discopygal.solvers.samplers import Sampler_Uniform, Sampler

from samplers.gauss_sampler import GaussSampler
from samplers.medial_sampler import MedialSampler
from samplers.sada_sampler import SadaSampler
from samplers.grid_sampler import GridSampler
from utils.experiment_wrapper import ExperimentsWrapper


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


def run_length_exp_algos(solver: str, sampler: Sampler = None, exact: bool = False):
    scenes = {
        'scene_length_1.json': (5, 1000, 0),
        'scene_length_2.json': (5, 1000, 0),
        'scene_length_3.json': (10, 3000, 0),
        'scene_length_4.json': (10, 1000, 0),
        'scene_length_5.json': (5, 1000, 2),
    }
    result = []  # list of lists: scene name, method name, avg time, avg path length
    num_experiments = 5

    for scene_name in scenes:
        # Get the parameters for the scene
        k, num_landmarks, bound = scenes[scene_name]
        string_printer.print(f'--------- Start Scene {scene_name}--------')
        with open(scene_name, 'r') as fp:
            curr_scene = Scene.from_dict(json.load(fp))

        experiment_wrapper = ExperimentsWrapper(curr_scene, solver, num_experiments=num_experiments,
                                                num_landmarks=num_landmarks, k=k,
                                                bounding_margin_width_factor=bound, sampler=sampler, exact=exact)
        time, path_len = experiment_wrapper.run()
        string_printer.print(
            f'Results for scene: {scene_name} for {num_experiments} experiments, for solver {solver} with {num_landmarks} samples, we have got {time:.5f} seconds and {path_len} path length')
    return


def length_k(scenes_path: list[str], solver: str, sampler: Sampler = None, num_experiments: int = 5,
             k_values: list[int] = [5, 15, 50], nearest_neighbors_metric: str = 'CTD',
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
                                                        time_limit=time_limit)
            elif solver == 'StaggeredGrid':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        nearest_neighbors_metric=nearest_neighbors_metric, eps=eps,
                                                        delta=delta,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        exact=exact, time_limit=time_limit)

            time, path_len = experiment_wrapper.run()
            string_printer.print(
                f'Results for k: {k}  we have got {time:.5f} seconds and {path_len} path length')
    return


def length_num_landmarks(scenes_path: list[str], solver: str, sampler: Sampler = None, num_experiments: int = 5,
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

            time, path_len = experiment_wrapper.run()
            string_printer.print(
                f'Results for num_landmark: {num_landmarks}  we have got {time:.5f} seconds and {path_len} path length')
    return


def length_metrics(scenes_path: list[str], solver: str, sampler: Sampler = None, num_experiments: int = 5,
                   k: int = 15,
                   nearest_neighbors_metrics: list[str] = ['', 'Euclidean', 'CTD', 'Epsilon_2', 'Epsilon_Inf', 'Max_L2',
                                                           'Mix_CTD', 'Mix_Epsilon_2'],
                   roadmap_nearest_neighbors_metric: str = ['', 'Euclidean', 'CTD', 'Epsilon_2', 'Epsilon_Inf',
                                                            'Max_L2', 'Mix_CTD', 'Mix_Epsilon_2'],
                   num_landmark: int = 1000, bound: FT = 0,
                   delta: int = 0.04, eps: int = 9999, prm_num_landmarks: int = 2000, exact: bool = False,
                   time_limit=100000):
    for scene_path in scenes_path:
        metrics_time_results = {}
        metrics_length_results = {}
        metrics_amount_of_runs_results= {}

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


def compare_algo(scenes_path: list[str], solvers: list[str], sampler: Sampler = None, num_experiments: int = 5,
                 k: int = 15, nearest_neighbors_metric: str = None, roadmap_nearest_neighbors_metric: str = None,
                 num_landmark: int = 5000, bound: FT = 0,
                 delta: int = 0.04, eps: int = 9999, prm_num_landmarks: int = 2000, exact: bool = False,
                 time_limit=100000):
    # Run experiments for each scene
    for scene_path in scenes_path:
        solvers_time_results = {}
        solvers_length_results = {}

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

            time, path_len = experiment_wrapper.run()
            solvers_length_results[solver] = path_len
            solvers_time_results[solver] = time

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
        string_printer.print(
            f'For {num_experiments} experiments, best time won solver {min_time_solver} with {min_time:.5f} seconds, best length won solver {min_len_solver} with {min_path_len}  path length')
        string_printer.print('_________________________')
    return


def get_scene_paths(scene_dir: str):
    scenes = []
    for root, dirs, files in os.walk("./scenes", topdown=False):
        for file in files:
            if file.endswith('.json'):
                scenes.append(os.path.join(root, file))
    return scenes


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser")

    # Add arguments
    parser.add_argument('--compare-landmarks', type=bool, default=False, help='Do landmarks experiments')
    parser.add_argument('--compare-algo', type=bool, default=False, help='Do algorithms experiments')
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
    parser.add_argument('--roadmap_nearest_neighbors', type=str, default=None,
                        choices=['CTD', 'Euclidean', 'Epsilon_2', 'Epsilon_Inf', 'Max_L2', 'Mix_CTD', 'Mix_Epsilon_2'],
                        help='Type of solver')
    parser.add_argument('--exact', type=bool, default=False, help='Run exact number of successful experiments')
    parser.add_argument('--path', type=str, default='./scenes/Easy2.json', help='Path to scene file')
    parser.add_argument('--sampler', type=str, default='none', choices=['none', 'uniform', 'combined'],
                        help='Type of sampler')
    parser.add_argument('--to-file', type=bool, default=False, help='Write output to file')
    parser.add_argument('--file', type=str, default='./results/other_benchmarks_tests.txt', help='Path to scene file')
    parser.add_argument('--append-to-file', type=bool, default=False, help='If file exist to append the output to him')
    parser.add_argument('--scene-dir', type=str, default='./scenes/', help='Path to scene directory')
    parser.add_argument('--time_limit', type=int, default=200, help='Second time limit')

    # Parse arguments
    args = parser.parse_args()
    return args


def start_running(args):
    with open(args.path, 'r') as fp:
        scene = Scene.from_dict(json.load(fp))

    # TODO: DELETE AFTER TESTS
    scenes = get_scene_paths(args.scene_dir)
    for scene in scenes:
        landmarks='dsaddasd'
        k ='dasdasdasd'
        if 'maze' in scene:
            landmarks = 3000
            k = 15
        else:
            continue

        length_metrics([scene], 'Squares', None, nearest_neighbors_metrics=['Epsilon_Inf'], k=k, num_landmark=landmarks, exact=True, num_experiments=10,
                       time_limit=20000)
    exit()

string_printer = StringPrinter()

if __name__ == '__main__':
    args = parse_arguments()
    args.compare_algo = True
    args.to_file = True
    args.append_to_file = True
    args.file = 'results/squares_only_euclaidean_20_precent.txt'
    string_printer.ready_printer(args)
    start_running(args)
