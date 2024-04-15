import math

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