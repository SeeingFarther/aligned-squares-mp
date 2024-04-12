import argparse
import json
from discopygal.solvers import Scene
from discopygal.bindings import *
from collections import defaultdict

from discopygal.solvers.samplers import Sampler_Uniform
from samplers.combined_sampler import CombinedSquaresSampler
from squares_planner import SquareMotionPlanner
from utils.experiment_wrapper import ExperimentsWrapper


def run_single_length_exp(scene_name, iterations, num_landmarks, k, margin):
    with open(scene_name, 'r') as fp:
        scene = Scene.from_dict(json.load(fp))
    method_times = defaultdict(list)
    method_lengths = defaultdict(list)
    curr_iteration = 0
    while curr_iteration < iterations:
        solver = SquareMotionPlanner(num_landmarks=num_landmarks, k=k, bounding_margin_width_factor=margin)
        solver.load_scene(scene)
        paths_len, times, names = solver.solve()  # Returns a PathCollection object
        if paths_len is None:
            continue
        for length, time_elapsed, method_name in zip(paths_len, times, names):
            method_times[method_name].append(time_elapsed)
            method_lengths[method_name].append(length)
        curr_iteration += 1
    result = []
    for name in method_lengths:
        result.append([scene_name, name, avg(method_times[name]), avg(method_lengths[name]), k, margin, num_landmarks])
    return result


def run_length_exp_algos():
    scenes = {
        'scene_length_1.json': (5, 1000, 0),
        'scene_length_2.json': (5, 1000, 0),
        'scene_length_3.json': (10, 3000, 0),
        'scene_length_4.json': (10, 1000, 0),
        'scene_length_5.json': (5, 1000, 2),
    }
    result = []  # list of lists: scene name, method name, avg time, avg path length
    iterations = 5
    for scene_name in scenes:
        k, num_landmarks, margin = scenes[scene_name]
        print(f'--------- Start Scene {scene_name}--------')
        scene_results = run_single_length_exp(scene_name, iterations, num_landmarks, k, margin)
        for scene_result in scene_results:
            result.append(scene_result)
    return result


def length_k():
    k_values = [5, 20, 50]
    num_landmarks = 1000
    margin = FT(0)
    scene_name = 'scene_length_1.json'
    result = []  # list of lists: scene name, method name, avg time, avg path length
    iterations = 5
    for k in k_values:
        scene_results = run_single_length_exp(scene_name, iterations, num_landmarks, k, margin)
        for scene_result in scene_results:
            result.append(scene_result)
    return result


def length_num_landmarks():
    k = 5
    num_landmarks_values = [500, 1000, 5000]
    margin = FT(0)
    scene_name = 'scene_length_1.json'
    result = []  # list of lists: scene name, method name, avg time, avg path length
    iterations = 5
    for num_landmarks in num_landmarks_values:
        scene_results = run_single_length_exp(scene_name, iterations, num_landmarks, k, margin)
        for scene_result in scene_results:
            result.append(scene_result)
    return result


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser")

    # Add arguments
    parser.add_argument('--k', type=int, default=15, help='Value of k')
    parser.add_argument('--num_landmarks', type=int, default=1000, help='Number of landmarks')
    parser.add_argument('-prm_num_landmarks', type=int, default=2000, help='Number of landmarks for PRM for DRRT')
    parser.add_argument('--num_experiments', type=int, default=5, help='Number of experiments')
    parser.add_argument('--bound', type=int, default=2, help='Bounding width factor')
    parser.add_argument('--eps', type=float, default=5, help='Number of experiments')
    parser.add_argument('--delta', type=float, default=2, help='Bounding width factor')
    parser.add_argument('--solver', type=str, default="squares", choices=['prm', 'drrt', 'staggered', 'squares'],
                        help='Type of solver')
    parser.add_argument('--path', type=str, default='scene_length_1.json', help='Path to scene file')
    parser.add_argument('--sampler', type=str, default='none', choices=['none', 'uniform', 'combined'],
                        help='Type of sampler')

    # Parse arguments
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    with open(args.path, 'r') as fp:
        scene = Scene.from_dict(json.load(fp))

    sampler = None
    if args.sampler == 'uniform':
        sampler = Sampler_Uniform()
    elif args.sampler == 'combined':
        sampler = CombinedSquaresSampler()

    experiment_wrapper = None
    if args.solver == 'prm':
        experiment_wrapper = ExperimentsWrapper(scene, 'PRM', num_experiments=args.num_experiments,
                                                num_landmarks=args.num_landmarks, k=args.k,
                                                bounding_margin_width_factor=args.bound, sampler=sampler)
    elif args.solver == 'drrt':
        experiment_wrapper = ExperimentsWrapper(scene, 'DRRT', num_experiments=args.num_experiments,
                                                prm_num_landmarks=args.prm_num_landmarks,
                                                num_landmarks=args.num_landmarks, k=args.k,
                                                bounding_margin_width_factor=args.bound, sampler=sampler)
    elif args.solver == 'staggered':
        experiment_wrapper = ExperimentsWrapper(scene, 'StaggeredGrid', num_experiments=args.num_experiments,
                                                eps=args.eps, delta=args.delta, bounding_margin_width_factor=args.bound,
                                                sampler=sampler)
    elif args.solver == 'squares':
        experiment_wrapper = ExperimentsWrapper(scene, 'Squares', num_experiments=args.num_experiments,
                                                num_landmarks=args.num_landmarks, k=args.k,
                                                bounding_margin_width_factor=args.bound, sampler=sampler)

    time, path_len = experiment_wrapper.run()
    print(
        f'Results for {args.num_experiments} experiments, for solver {args.solver} we have got {time:.3f} seconds and {path_len} path length')

# if __name__ == '__main__':
# print("\n\n\nrun_length_exp_algos\n")
# for stat in run_length_exp_algos():
#     print(stat)
# print("\n\n\nlength_k\n")
# for stat in length_k():
#     print(stat)
# print("\n\n\nlength_num_landmarks\n")
# for stat in length_num_landmarks():
#     print(stat)
