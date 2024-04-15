import math
import random

import numpy as np
from scipy.optimize import minimize
import sklearn.metrics
from discopygal.bindings import *
from discopygal.solvers.metrics import Metric
from scipy.spatial.distance import cdist

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


class MetricNotImplemented(Exception):
    pass


class Metric_Epsilon(Metric):
    @staticmethod
    def l2_distance(p1, p2):
        """
        Calculates the Euclidean distance between two points.
        """
        return np.linalg.norm(np.array(p1) - np.array(p2))

    @staticmethod
    def l_inf_distance(p1, p2):
        """
        Calculates the maximum distance between two points.
        """
        return np.max(np.abs(np.array(p1) - np.array(p2)))

    @staticmethod
    def congruence_objective(translation, start_robots_points, end_robots_points, metric):
        """
        Objective function for the ε-congruence minimization problem.
        """
        start_robots_points = np.array(start_robots_points)
        end_robots_points = np.array(end_robots_points)
        translated_start_robots_points = start_robots_points + np.array(translation)
        distances = cdist(translated_start_robots_points, end_robots_points, metric=metric)
        diagonal = np.diagonal(distances)
        return np.max(diagonal)

    @staticmethod
    def find_optimal_translation(start_robots_points, end_robots_points, metric):
        """
        Finds the optimal translation to minimize the ε-congruence between the two robots.
        """
        initial_guess = (0, 0)
        result = minimize(Metric_Epsilon.congruence_objective, initial_guess,
                          args=(start_robots_points, end_robots_points, metric), method='Powell')
        min_epsilon = result.fun
        return min_epsilon

    @staticmethod
    def create_arrays(length, p, q):
        curr = []
        next = []
        for i in range(length):
            curr.append((p[2 * i], p[2 * i + 1]))
            next.append((q[2 * i], q[2 * i + 1]))
        return curr, next

    @staticmethod
    def points_to_list(p, q):
        length = int(len(p) / 2)
        return Metric_Epsilon.create_arrays(length, p, q)

    @staticmethod
    def points_d_to_list(p, q):
        length = int(p.dimension() / 2)
        return Metric_Epsilon.create_arrays(length, p, q)


class Metric_Epsilon_2(Metric_Epsilon):
    """
    Implementation of the Euclidean metric for nearest neighbors search
    """

    @staticmethod
    def dist(p, q):
        """
        Return the distance between two points

        :param p: first point
        :type p: :class:`~discopygal.bindings.Point_2` or :class:`~discopygal.bindings.Point_d`
        :param q: second point
        :type q: :class:`~discopygal.bindings.Point_2` or :class:`~discopygal.bindings.Point_d`

        :return: distance between p and q
        :rtype: :class:`~discopygal.bindings.FT`
        """

        if type(p) is Point_2 and type(q) is Point_2:
            curr, next = [p.x().to_double(), p.y().to_double()], [q.x().to_double(), q.y().to_double()]
            epsilon_2 = Metric_Epsilon_2.find_optimal_translation(curr, next, Metric_Epsilon.l2_distance)
            return FT(epsilon_2)
        elif type(p) is Point_d and type(
                q) is Point_d and p.dimension() == q.dimension() and p.dimension() % 2 == 0 and q.dimension() % 2 == 0:
            curr, next = Metric_Epsilon_2.points_d_to_list(p, q)
            epsilon_2 = Metric_Epsilon_2.find_optimal_translation(curr, next, Metric_Epsilon.l2_distance)
            return FT(epsilon_2)
        else:
            raise MetricNotImplemented('p,q should be Point_2 or Point_d or Not even dimension')

    @staticmethod
    def float_dist(p, q):
        """
        Return the distance between two points

        :param p: first point
        :type p: :class:`~discopygal.bindings.Point_2` or :class:`~discopygal.bindings.Point_d`
        :param q: second point
        :type q: :class:`~discopygal.bindings.Point_2` or :class:`~discopygal.bindings.Point_d`

        :return: distance between p and q
        :rtype: :class:float
        """
        if len(p) == len(q) and len(p) % 2 == 0 and len(q) % 2 == 0:
            curr, next = Metric_Epsilon_2.points_to_list(p, q)
            epsilon_2 = Metric_Epsilon_2.find_optimal_translation(curr, next, Metric_Epsilon.l2_distance)
            return epsilon_2
        else:
            raise MetricNotImplemented('p,q should be Point_2 or Point_d or Not even dimension')

    @staticmethod
    def sklearn_impl():
        """
        Return the metric as sklearn metric object.
        """
        # Implementation specific to scikit-learn
        return sklearn.metrics.DistanceMetric.get_metric(metric='pyfunc', func=Metric_Epsilon_2.float_dist)


class Metric_Epsilon_Inf(Metric_Epsilon):
    """
    Implementation of the Euclidean metric for nearest neighbors search
    """

    @staticmethod
    def dist(p, q):
        """
        Return the distance between two points

        :param p: first point
        :type p: :class:`~discopygal.bindings.Point_2` or :class:`~discopygal.bindings.Point_d`
        :param q: second point
        :type q: :class:`~discopygal.bindings.Point_2` or :class:`~discopygal.bindings.Point_d`

        :return: distance between p and q
        :rtype: :class:`~discopygal.bindings.FT`
        """
        if type(p) is Point_2 and type(q) is Point_2:
            curr, next = [p.x().to_double(), p.y().to_double()], [q.x().to_double(), q.y().to_double()]
            epsilon_inf = Metric_Epsilon_Inf.find_optimal_translation(curr, next, Metric_Epsilon.l_inf_distance)
            return FT(epsilon_inf)
        elif type(p) is Point_d and type(
                q) is Point_d and p.dimension() == q.dimension() and p.dimension() % 2 == 0 and q.dimension() % 2 == 0:
            curr, next = Metric_Epsilon_Inf.points_d_to_list(p, q)
            epsilon_inf = Metric_Epsilon_Inf.find_optimal_translation(curr, next, Metric_Epsilon.l_inf_distance)
            return FT(epsilon_inf)
        else:
            raise MetricNotImplemented('p,q should be Point_2 or Point_d or Not even dimension')

    @staticmethod
    def float_dist(p, q):
        """
        Return the distance between two points

        :param p: first point
        :type p: :class:`~discopygal.bindings.Point_2` or :class:`~discopygal.bindings.Point_d`
        :param q: second point
        :type q: :class:`~discopygal.bindings.Point_2` or :class:`~discopygal.bindings.Point_d`

        :return: distance between p and q
        :rtype: :class:float
        """
        if len(p) == len(q) and len(p) % 2 == 0 and len(q) % 2 == 0:
            curr, next = Metric_Epsilon_Inf.points_to_list(p, q)
            epsilon_inf = Metric_Epsilon_Inf.find_optimal_translation(curr, next, Metric_Epsilon.l_inf_distance)
            return epsilon_inf
        else:
            raise MetricNotImplemented('p,q should be Point_2 or Point_d or Not even dimension')

    @staticmethod
    def sklearn_impl():
        """
        Return the metric as sklearn metric object.
        """
        # Implementation specific to scikit-learn
        return sklearn.metrics.DistanceMetric.get_metric(metric='pyfunc', func=Metric_Epsilon_Inf.float_dist)


import numpy as np
from scipy.optimize import minimize


def calculate_min_max_distance(start_robots_points, end_robots_points, translation):
    """
    Calculates the minimum of the maximum distances between corresponding points after applying the translation.
    """
    translated_start_robots_points = [(start_point[0] + translation[0], start_point[1] + translation[1]) for start_point
                                      in start_robots_points]
    distances = [np.linalg.norm(np.array(translated_point) - np.array(end_point)) for translated_point, end_point in
                 zip(translated_start_robots_points, end_robots_points)]
    return np.max(distances)


# Function to compute the translations using the equations
def compute_translation(U, V):
    m = len(U)
    dx = -np.sum(U[:, 0] - V[:, 0]) / m
    dy = -np.sum(U[:, 1] - V[:, 1]) / m
    return dx, dy


# Objective function for the ε-congruence minimization problem
def congruence_objective(translation, U, V):
    translated_U = U + np.array(translation)
    distances = np.linalg.norm(translated_U - V, axis=1)
    return np.max(distances)


# def linf_congruence_exact(U, V):
#     """
#     Calculates the exact solution for Linf metric ε-congruence.
#
#     Args:
#     - U: List of tuples representing the coordinates of points in the first set
#     - V: List of tuples representing the coordinates of points in the second set
#
#     Returns:
#     - Tx: Translation in the x-direction
#     - Ty: Translation in the y-direction
#     """
#     # Calculate the maximum absolute difference in x and y coordinates
#     max_diff_x = max(abs(ui[0] - vi[0]) for ui, vi in zip(U, V))
#     max_diff_y = max(abs(ui[1] - vi[1]) for ui, vi in zip(U, V))
#
#     # Calculate the translation in the x and y directions
#     Tx = -max_diff_x
#     Ty = -max_diff_y
#
#     return Tx, Ty
#
# def Linf_objective(translation, start_points, end_points):
#     """
#     Objective function for the ε-congruence minimization problem with Linf distance metric.
#     """
#     translated_points = start_points + np.array(translation)
#     distances = np.max(np.abs(translated_points - end_points), axis=1)
#     return np.max(distances)  # Maximize the maximum distance
#
# def find_optimal_translation_Linf(start_points, end_points):
#     """
#     Finds the optimal translation to minimize the ε-congruence with Linf distance metric.
#     """
#     initial_guess = (0, 0)
#     result = minimize(Linf_objective, initial_guess, args=(start_points, end_points), method='Powell')
#     min_epsilon = result.x
#     return min_epsilon
#
# # Example 1
# start_robots_points_1 = np.array([(0, 2), (1, 2)])
# end_robots_points_1 = np.array([(3, 4), (2, 1)])
# optimal_translation_1 = find_optimal_translation_Linf(start_robots_points_1, end_robots_points_1)
# exact_translation = linf_congruence_exact(start_robots_points_1, end_robots_points_1)
#
# # Example 2
# start_robots_points_2 = np.array([(2, 3), (4, 5)])
# end_robots_points_2 = np.array([(1, 1), (0, 0)])
# optimal_translation_2 = find_optimal_translation_Linf(start_robots_points_2, end_robots_points_2)
# exact_translation = linf_congruence_exact(start_robots_points_2, end_robots_points_2)
#
# print("Example 1:")
# print("Optimal translation for L2inf distance metric:", optimal_translation_1)
# print("Exact translation for L∞ distance metric:", exact_translation)
#
#
# print("\nExample 2:")
# print("Optimal translation for L2inf distance metric:", optimal_translation_2)
# print("Exact translation for L∞ distance metric:", exact_translation)
def epsilon_congruence_linf1(U, V):
    """
    Compute the epsilon-congruence with respect to the L_inf metric and find the optimal translation.

    Args:
        U (np.ndarray): Array of shape (m, n) representing the joint configurations (u_1, ..., u_m)
        V (np.ndarray): Array of shape (m, n) representing the joint configurations (v_1, ..., v_m)

    Returns:
        float: The epsilon-congruence with respect to the L_inf metric
        np.ndarray: The optimal translation vector T^*
    """
    m, n = U.shape
    diffs = V - U  # Compute the vectors d_i = v_i - u_i

    # Find the median components of d_i across all i = 1, ..., m
    median_components = np.median(diffs, axis=0)

    # Define the optimal translation T^*
    optimal_translation = median_components

    # Compute the translated U with the optimal translation
    translated_U = U + optimal_translation

    # Compute the epsilon-congruence with respect to L_inf
    epsilon_linf = np.max(np.abs(translated_U - V), axis=1).max()

    return epsilon_linf, optimal_translation


def linf_congruence_optimal(start_robots_points, end_robots_points):
    """
    Finds the optimal translation to minimize the Linf metric ε-congruence between two sets of points.
    """
    initial_guess = (0, 0)
    result = minimize(linf_congruence_objective, initial_guess, args=(start_robots_points, end_robots_points),
                      method='Powell')
    min_epsilon = result.fun
    return min_epsilon, result.x[0], result.x[1]


def linf_congruence_objective(translation, start_robots_points, end_robots_points):
    """
    Objective function for the Linf metric ε-congruence minimization problem.
    """
    translated_start_points = start_robots_points + np.array(translation)
    distances = np.max(np.abs(translated_start_points - end_robots_points), axis=1)
    return np.max(distances)


def linf_congruence_exact(start_robots_points, end_robots_points):
    """
    Calculates the exact solution for Linf metric ε-congruence.
    """
    max_diff_x = max(abs(ui[0] - vi[0]) for ui, vi in zip(start_robots_points, end_robots_points))
    max_diff_y = max(abs(ui[1] - vi[1]) for ui, vi in zip(start_robots_points, end_robots_points))
    Tx = max_diff_x
    Ty = max_diff_y
    return Tx, Ty


# Example data
start_robots_points_1 = np.array([(0, 2), (1, 2)])
end_robots_points_1 = np.array([(3, 4), (2, 1)])

# Find optimal translation using scipy optimization
min_epsilon_optimal, Tx_optimal, Ty_optimal = linf_congruence_optimal(start_robots_points_1, end_robots_points_1)

# Find exact solution
Tx_exact, Ty_exact = Metric_Epsilon_Inf.find_optimal_translation(start_robots_points_1, end_robots_points_1,
                                                                 Metric_Epsilon.l_inf_distance), 0

Tx_exact1, Ty_exact1 = linf_congruence_exact(start_robots_points_1, end_robots_points_1)

# Print results
# print("Optimal solution:")
# print("Minimum Linf ε-congruence:", min_epsilon_optimal)
# print("Translation (Tx, Ty):", (Tx_optimal, Ty_optimal))
# print("\nExact solution:")
# print("Translation (Tx, Ty):", (Tx_exact, Ty_exact))
# print("Translation (Tx1, Ty1):", (Tx_exact1, Ty_exact1))
# print("Translation (Tx1, Ty1):", epsilon_congruence_linf(start_robots_points_1, end_robots_points_1))

print()
start_robots_points_1 = np.array([(0, 22), (115, 22)])
end_robots_points_1 = np.array([(32, 42), (2, 21)])

# Find optimal translation using scipy optimization
min_epsilon_optimal, Tx_optimal, Ty_optimal = linf_congruence_optimal(start_robots_points_1, end_robots_points_1)

# Find exact solution
Tx_exact, Ty_exact = Metric_Epsilon_Inf.find_optimal_translation(start_robots_points_1, end_robots_points_1,
                                                                 Metric_Epsilon.l_inf_distance), 0

# Print results
# print("Optimal solution:")
# print("Minimum Linf ε-congruence:", min_epsilon_optimal)
# print("Translation (Tx, Ty):", (Tx_optimal, Ty_optimal))
# print("\nExact solution:")
# print("Translation (Tx, Ty):", (Tx_exact, Ty_exact))
# print("Translation (Tx1, Ty1):", epsilon_congruence_linf(start_robots_points_1, end_robots_points_1))

count = 0
for i in range(5000):
    start_robots_points_1 = np.array([(random.random() * 500, random.random() * 500), (random.random() * 500, random.random() * 500)])
    end_robots_points_1 = np.array([(random.random() * 500, 42), (random.random() * 500, random.random() * 500)])

    # start_robots_points_1 = np.array([[41.00100235, 71.24031091], [97.09869774, 252.60733371]])
    # end_robots_points_1 = np.array([[441.10610708, 42.0], [248.26573763, 20.22851399]])
    # Find optimal translation using scipy optimization
    min_epsilon_optimal, Tx_optimal, Ty_optimal = linf_congruence_optimal(start_robots_points_1, end_robots_points_1)
    min_epsilon_optimal1, Texact = epsilon_congruence_linf1(start_robots_points_1, end_robots_points_1)

    # Find exact solution
    Tx_exact, Ty_exact = Metric_Epsilon_Inf.find_optimal_translation(start_robots_points_1, end_robots_points_1,
                                                                     Metric_Epsilon.l_inf_distance), 0
    if abs(abs(min_epsilon_optimal) - abs(min_epsilon_optimal1)) > 0.5:
        # print(start_robots_points_1)
        # print(end_robots_points_1)
        # Print results
        # print("Optimal solution:")
        # print("Minimum Linf ε-congruence:", min_epsilon_optimal)
        # print("Translation (Tx, Ty):", (Tx_optimal, Ty_optimal))
        # print("Translation (Tx, Ty):", (Tx_exact, Ty_exact))
        # print("\nExact solution:")
        # print("Translation (Tx1, Ty1):", (min_epsilon_optimal1, Texact))
        count +=1

print(count)