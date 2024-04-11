from discopygal.solvers.prm import PRM
import json
import time
from discopygal.solvers import Scene, Path, PathCollection
from typing import Optional
from discopygal.bindings import *
from discopygal.geometry_utils import conversions
from collections import defaultdict
from squares_planner import SquareMotionPlanner




# class BasicPrmForExperiments(PRM):
#     def load_scene(self, scene: Scene):
#         super().load_scene(scene)
#         self.obstacles = scene.obstacles
#
#     def solve(self):
#         path = super().solve()
#         if not path.paths:
#             clearance = 0
#         else:
#             # compute_path_clearance(path: Path, min_y: float, max_y: float, obstacles
#             clearance = compute_path_clearance(path, self._bounding_box.min_y.to_double(), self._bounding_box.max_y.to_double(), self.obstacles)
#         return clearance, path
#


def avg(lst: list):
    if not lst:
        return 0
    return sum(lst) / len(lst)


def run_single_length_exp(scene_name, iterations, num_landmarks, k, margin):
    with open(scene_name, 'r') as fp:
        scene = Scene.from_dict(json.load(fp))
    method_times = defaultdict(list)
    method_lengths = defaultdict(list)
    curr_iteration = 0
    while (curr_iteration < iterations):
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


if __name__ == '__main__':
    # print("\n\n\nrun_length_exp_algos\n")
    # for stat in run_length_exp_algos():
    #     print(stat)
    # print("\n\n\nlength_k\n")
    # for stat in length_k():
    #     print(stat)
    # print("\n\n\nlength_num_landmarks\n")
    # for stat in length_num_landmarks():
        #print(stat)
