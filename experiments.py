import argparse
import json
import os
from contextlib import redirect_stdout

from discopygal.solvers import Scene
from discopygal.bindings import *
from discopygal.solvers.samplers import Sampler_Uniform, Sampler

from samplers.bridge_sampler import BridgeSampler
from samplers.combined_sampler import CombinedSampler
from samplers.middle_sampler import MiddleSampler
from samplers.space_sampler import SpaceSampler
from utils.experiment_wrapper import ExperimentsWrapper


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
        print(f'--------- Start Scene {scene_name}--------')
        with open(scene_name, 'r') as fp:
            curr_scene = Scene.from_dict(json.load(fp))

        experiment_wrapper = ExperimentsWrapper(curr_scene, solver, num_experiments=num_experiments,
                                                num_landmarks=num_landmarks, k=k,
                                                bounding_margin_width_factor=bound, sampler=sampler, exact=exact)
        time, path_len = experiment_wrapper.run()
        print(
            f'Results for scene: {scene_name} for {num_experiments} experiments, for solver {solver} with {num_landmarks} samples, we have got {time:.5f} seconds and {path_len} path length')
    return


def length_k(scenes_path: list[str], solver: str, sampler: Sampler = None, num_experiments: int = 5,
             k_values: list[int] = [5, 15, 50],
             num_landmark: int = 1000, bound: FT = 0, delta: int = 0.1, eps: int = 9999,
             prm_num_landmarks: int = 2000, exact: bool = False, time_limit=100000):
    # Run experiments for each scene
    for scene_path in scenes_path:
        scene_name = scene_path.split('/')[-1].split('.')[0]
        with open(scene_path, 'r') as fp:
            scene = Scene.from_dict(json.load(fp))

        print("_________________________")
        print("Results for scene: ", scene_name)
        print("For following parameters:")
        print("Solver: ", solver)
        print("Number of experiments: ", num_experiments)
        print("num_landmark: ", num_landmark)
        print("Bounding width factor: ", bound)
        print("Delta: ", delta)
        print("Epsilon: ", eps)
        print("PRM Number of landmarks: ", prm_num_landmarks)
        print("Exact: ", exact)

        for k in k_values:
            if solver == 'PRM' or solver == 'Squares':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmark, k=k,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        exact=exact, time_limit=time_limit)
            elif solver == 'DRRT':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmark, k=k,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        prm_num_landmarks=prm_num_landmarks, exact=exact,
                                                        time_limit=time_limit)
            elif solver == 'StaggeredGrid':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        eps=eps, delta=delta,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        exact=exact, time_limit=time_limit)

            time, path_len = experiment_wrapper.run()
            print(
                f'Results for k: {k}  we have got {time:.5f} seconds and {path_len} path length')
    return


def length_num_landmarks(scenes_path: list[str], solver: str, sampler: Sampler = None, num_experiments: int = 5,
                         k: int = 15, num_landmarks_values: list[int] = [500, 1000, 5000], bound: FT = 0,
                         delta: int = 0.04, eps: int = 9999, prm_num_landmarks: int = 2000, exact: bool = False,
                         time_limit=100000):
    for scene_path in scenes_path:
        scene_name = scene_path.split('/')[-1].split('.')[0]
        with open(scene_path, 'r') as fp:
            scene = Scene.from_dict(json.load(fp))

        print("_________________________")
        print("Results for scene: ", scene_name)
        print("For following parameters:")
        print("Solver: ", solver)
        print("Number of experiments: ", num_experiments)
        print("K: ", k)
        print("Bounding width factor: ", bound)
        print("Delta: ", delta)
        print("Epsilon: ", eps)
        print("PRM Number of landmarks: ", prm_num_landmarks)
        print("Exact: ", exact)

        for num_landmarks in num_landmarks_values:
            if solver == 'PRM' or solver == 'Squares':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmarks, k=k,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        exact=exact, time_limit=time_limit)
            elif solver == 'DRRT':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmarks, k=k,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        prm_num_landmarks=prm_num_landmarks, exact=exact,
                                                        time_limit=time_limit)
            elif solver == 'StaggeredGrid':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        eps=eps, delta=delta,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        exact=exact, time_limit=time_limit)

            time, path_len = experiment_wrapper.run()
            print(
                f'Results for num_landmark: {num_landmarks}  we have got {time:.5f} seconds and {path_len} path length')
    return


def compare_algo(scenes_path: list[str], solvers: list[str], sampler: Sampler = None, num_experiments: int = 5,
                 k: int = 15, num_landmark: int = 5000, bound: FT = 0,
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
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        exact=exact, time_limit=time_limit)
            elif solver == 'DRRT':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmark, k=k,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        prm_num_landmarks=prm_num_landmarks, exact=exact,
                                                        time_limit=time_limit)
            elif solver == 'StaggeredGrid':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        eps=eps, delta=delta,
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

        print("_________________________")
        print("Results for scene: ", scene_name)
        print("For following parameters:")
        print("Solver: ", solvers)
        print("Number of experiments: ", num_experiments)
        print("K: ", k)
        print("Number of landmarks: ", num_landmark)
        print("Bounding width factor: ", bound)
        print("Delta: ", delta)
        print("Epsilon: ", eps)
        print("PRM Number of landmarks: ", prm_num_landmarks)
        print("Exact: ", exact)
        print("Time Limit: ", time_limit)
        print("Times: ", solvers_time_results)
        print("Path Lengths: ", solvers_length_results)
        print(
            f'For {num_experiments} experiments, best time won solver {min_time_solver} with {min_time:.5f} seconds, best length won solver {min_len_solver} with {min_path_len}  path length')
        print("_________________________")
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
    parser.add_argument('--compare-length', type=bool, default=False, help='Do length experiments')
    parser.add_argument('--k', type=int, default=15, help='Value of k')
    parser.add_argument('--num-landmarks', type=int, default=1000, help='Number of landmarks')
    parser.add_argument('-prm-num-landmarks', type=int, default=2000, help='Number of landmarks for PRM for DRRT')
    parser.add_argument('--num_experiments', type=int, default=5, help='Number of experiments')
    parser.add_argument('--bound', type=int, default=2, help='Bounding width factor')
    parser.add_argument('--eps', type=float, default=5, help='Number of experiments')
    parser.add_argument('--delta', type=float, default=2, help='Bounding width factor')
    parser.add_argument('--solver', type=str, default="squares", choices=['prm', 'drrt', 'staggered', 'squares'],
                        help='Type of solver')
    parser.add_argument('--metric', type=str, default=None, choices=['CTD', 'Euclidean', 'Epsilon_2', 'Epsilon_Inf'],
                        help='Type of solver')
    parser.add_argument('--exact', type=bool, default=False, help='Run exact number of successful experiments')
    parser.add_argument('--path', type=str, default='./scenes/easy2.json', help='Path to scene file')
    parser.add_argument('--sampler', type=str, default='none', choices=['none', 'uniform', 'combined'],
                        help='Type of sampler')
    parser.add_argument('--to-file', type=bool, default=False, help='Write output to file')
    parser.add_argument('--file', type=str, default='./results/test', help='Path to scene file')
    parser.add_argument('--scene-dir', type=str, default='./scenes/', help='Path to scene directory')
    parser.add_argument('--time_limit', type=int, default=200, help='Second time limit')

    # Parse arguments
    args = parser.parse_args()
    return args


def start_running(args):
    with open(args.path, 'r') as fp:
        scene = Scene.from_dict(json.load(fp))

    if args.compare_length:
        scenes = get_scene_paths(args.scene_dir)

        # compare_algo(scenes, ['PRM', 'DRRT', 'StaggeredGrid', 'Squares'], None)
        compare_algo(scenes, ['Squares'], num_landmark=1500, sampler=None, exact=True, time_limit=600)
        # compare_algo(scenes, ['PRM', 'DRRT', 'StaggeredGrid', 'Squares'], None, num_experiments=args.num_experiments, k=args.k, num_landmark=args.num_landmarks, bound=args.bound, delta=args.delta, eps=args.eps, prm_num_landmarks=args.prm_num_landmarks)
        exit(0)

    if args.compare_landmarks:
        scenes = get_scene_paths(args.scene_dir)
        length_num_landmarks(scenes, solver=args.solver, num_experiments=args.num_experiments, bound=args.bound, k=args.k,
                 delta=args.delta, eps=args.eps, prm_num_landmarks=args.prm_num_landmarks, exact=args.exact,
                 time_limit=args.time_limit)
        exit(0)

    if args.compare_k:
        scenes = get_scene_paths(args.scene_dir)
        length_k(scenes, solver=args.solver, num_experiments=args.num_experiments, bound=args.bound,
                 num_landmark=args.num_landmarks, delta=args.delta, eps=args.eps,
                 prm_num_landmarks=args.prm_num_landmarks, exact=args.exact, time_limit=args.time_limit)
        exit(0)

    sampler = None
    if args.sampler == 'uniform':
        sampler = Sampler_Uniform()
    elif args.sampler == 'combined':
        samplers = [SpaceSampler(), BridgeSampler(), MiddleSampler(y_axis=True), MiddleSampler(y_axis=False)]
        sampler = CombinedSampler(samplers)

    experiment_wrapper = None
    if args.solver == 'prm':
        experiment_wrapper = ExperimentsWrapper(scene, 'PRM', num_experiments=args.num_experiments,
                                                num_landmarks=args.num_landmarks, k=args.k,
                                                bounding_margin_width_factor=args.bound, sampler=sampler,
                                                metric=args.metric)
    elif args.solver == 'drrt':
        experiment_wrapper = ExperimentsWrapper(scene, 'DRRT', num_experiments=args.num_experiments,
                                                prm_num_landmarks=args.prm_num_landmarks,
                                                num_landmarks=args.num_landmarks, k=args.k,
                                                bounding_margin_width_factor=args.bound, sampler=sampler,
                                                metric=args.metric)
    elif args.solver == 'staggered':
        experiment_wrapper = ExperimentsWrapper(scene, 'StaggeredGrid', num_experiments=args.num_experiments,
                                                eps=args.eps, delta=args.delta, bounding_margin_width_factor=args.bound,
                                                sampler=sampler, metric=args.metric)
    elif args.solver == 'squares':
        experiment_wrapper = ExperimentsWrapper(scene, 'Squares', num_experiments=args.num_experiments,
                                                num_landmarks=args.num_landmarks, k=args.k,
                                                bounding_margin_width_factor=args.bound, sampler=sampler,
                                                metric=args.metric)

    time_result, path_len_result = experiment_wrapper.run()
    print(
        f'Results for {args.num_experiments} experiments, for solver {args.solver} we have got {time_result:.3f} seconds and {path_len_result} path length')


if __name__ == '__main__':
    args = parse_arguments()
    args.compare_length = True
    args.to_file = True
    args.file = './results/Length_Experiments2.txt'

    if args.to_file:
        with open(args.file, 'w') as f:
            with redirect_stdout(f):
                start_running(args)
    else:
        start_running(args)
