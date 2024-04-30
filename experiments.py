import argparse
import json
import os

from discopygal.solvers import Scene
from discopygal.bindings import *
from discopygal.solvers.samplers import Sampler_Uniform, Sampler

from samplers.combined_sampler import CombinedSampler
from utils.experiment_wrapper import ExperimentsWrapper


def run_length_exp_algos(solver: str, sampler: Sampler = None):
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
                                                bounding_margin_width_factor=bound, sampler=sampler)
        time, path_len = experiment_wrapper.run()
        print(
            f'Results for scene: {scene_name} for {num_experiments} experiments, for solver {solver} with {num_landmarks} samples, we have got {time:.3f} seconds and {path_len} path length')
    return


def length_k(scenes_path: list[str], solver: str, sampler: Sampler = None, num_experiments: int = 5,
             k_values: list[int] = [5, 15, 50],
             num_landmark: int = 1000, bound: FT = 0, delta: int = 0.1, eps: int = 9999,
             prm_num_landmarks: int = 2000):

    # Run experiments for each scene
    for scene_path in scenes_path:
        scene_name = scene_path.split('/')[-1].split('.')[0]
        with open(scene_path, 'r') as fp:
            scene = Scene.from_dict(json.load(fp))

        for k in k_values:
            if solver == 'PRM' or solver == 'Squares':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmark, k=k,
                                                        bounding_margin_width_factor=bound, sampler=sampler)
            elif solver == 'DRRT':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmark, k=k,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        prm_num_landmarks=prm_num_landmarks)
            elif solver == 'StaggeredGrid':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        eps=eps, delta=delta,
                                                        bounding_margin_width_factor=bound, sampler=sampler)

            time, path_len = experiment_wrapper.run()
            print(
                f'Results for scene: {scene_name} for {num_experiments} experiments, for solver {solver} we have got {time:.3f} seconds and {path_len} path length')
    return


def length_num_landmarks(scenes_path: list[str], solver: str, sampler: Sampler = None, num_experiments: int = 5,
                         k: int = 15, num_landmarks_values: list[int] = [500, 1000, 5000], bound: FT = 0,
                         delta: int = 0.1, eps: int = 9999, prm_num_landmarks: int = 2000):
    for scene_path in scenes_path:
        scene_name = scene_path.split('/')[-1].split('.')[0]
        with open(scene_path, 'r') as fp:
            scene = Scene.from_dict(json.load(fp))

        for num_landmarks in num_landmarks_values:
            if solver == 'PRM' or solver == 'Squares':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmarks, k=k,
                                                        bounding_margin_width_factor=bound, sampler=sampler)
            elif solver == 'DRRT':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        num_landmarks=num_landmarks, k=k,
                                                        bounding_margin_width_factor=bound, sampler=sampler,
                                                        prm_num_landmarks=prm_num_landmarks)
            elif solver == 'StaggeredGrid':
                experiment_wrapper = ExperimentsWrapper(scene, solver, num_experiments=num_experiments,
                                                        eps=eps, delta=delta,
                                                        bounding_margin_width_factor=bound, sampler=sampler)

            time, path_len = experiment_wrapper.run()
            print(
                f'Results for scene: {scene_name} for {num_experiments} experiments, for solver {solver} we have got {time:.3f} seconds and {path_len} path length')
    return


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
    parser.add_argument('--path', type=str, default='scene_length_2.json', help='Path to scene file')
    parser.add_argument('--sampler', type=str, default='none', choices=['none', 'uniform', 'combined'],
                        help='Type of sampler')

    # Parse arguments
    args = parser.parse_args()
    return args


print('Running experiments for length of path')
# print('Running experiments for PRM')
# length_num_landmarks(['scene_length_1.json', './scenes/bug_trap1.json', './scenes/cubic3.json'], 'PRM',None)
# print('Running experiments for DRRT')
# length_num_landmarks(['scene_length_1.json', './scenes/bug_trap1.json', './scenes/cubic3.json'], 'DRRT', None)
# print('Running experiments for StaggeredGrid')
# length_num_landmarks(['scene_length_1.json', './scenes/bug_trap1.json', './scenes/cubic3.json'], 'StaggeredGrid',None)
print('Running experiments for Squares')

scenes = []
for root, dirs, files in os.walk("./scenes", topdown=False):
    for file in files:
        if file.endswith('.json'):
            scenes.append(os.path.join(root, file))

length_num_landmarks(scenes, 'Squares', None)

if __name__ == '__main__':
    args = parse_arguments()
    with open(args.path, 'r') as fp:
        scene = Scene.from_dict(json.load(fp))

    sampler = None
    if args.sampler == 'uniform':
        sampler = Sampler_Uniform()
    elif args.sampler == 'combined':
        sampler = CombinedSampler()

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

    time_result, path_len_result = experiment_wrapper.run()
    print(
        f'Results for {args.num_experiments} experiments, for solver {args.solver} we have got {time_result:.3f} seconds and {path_len_result} path length')
