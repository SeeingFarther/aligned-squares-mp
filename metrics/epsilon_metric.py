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
    def compute_translation_l2(U, V):
        m = len(U)
        dx = -np.sum(U[:, 0] - V[:, 0]) / m
        dy = -np.sum(U[:, 1] - V[:, 1]) / m
        return dx, dy

    @staticmethod
    def congruence_objective_l2(U, V):
        translation = Metric_Epsilon.compute_translation_l2(U, V)
        translated_start_robots_points = [(start_point[0] + translation[0], start_point[1] + translation[1]) for
                                          start_point in U]
        distances = [np.linalg.norm(np.array(translated_point) - np.array(end_point)) for translated_point, end_point in
                     zip(translated_start_robots_points, V)]
        return np.max(distances)

    @staticmethod
    def compute_translation_l_inf(U, V):
        diffs = V - U  # Compute the vectors d_i = v_i - u_i

        # Find the median components of d_i across all i = 1, ..., m
        median_components = np.median(diffs, axis=0)

        # Define the optimal translation T^*
        optimal_translation = median_components

        return optimal_translation

    @staticmethod
    def congruence_objective_l_inf(U, V):
        translation = Metric_Epsilon.compute_translation_l_inf(U, V)

        # Compute the epsilon-congruence with respect to L_inf
        epsilon_l_inf = np.max(np.abs(translation - V), axis=1).max()

        return epsilon_l_inf

    @staticmethod
    def create_arrays(length, p, q):
        curr = [(p[2 * i], p[2 * i + 1]) for i in range(length)]
        next = [(q[2 * i], q[2 * i + 1]) for i in range(length)]
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
            U, V = np.array([p.x().to_double(), p.y().to_double()]), np.array([q.x().to_double(), q.y().to_double()])
            epsilon_2 = Metric_Epsilon_2.congruence_objective_l2(U, V)
            return FT(epsilon_2)
        elif type(p) is Point_d and type(
                q) is Point_d and p.dimension() == q.dimension() and p.dimension() % 2 == 0 and q.dimension() % 2 == 0:
            U, V = Metric_Epsilon_2.points_d_to_list(p, q)
            U, V = np.array(U), np.array(V)
            epsilon_2 = Metric_Epsilon_2.congruence_objective_l2(U, V)
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
            U, V = Metric_Epsilon_2.points_to_list(p, q)
            U, V = np.array(U), np.array(V)
            epsilon_2 = Metric_Epsilon_2.congruence_objective_l2(U, V)
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
            U, V = np.array([p.x().to_double(), p.y().to_double()]), np.array([q.x().to_double(), q.y().to_double()])
            epsilon_inf = Metric_Epsilon_Inf.congruence_objective_l_inf(U, V)
            return FT(epsilon_inf)
        elif type(p) is Point_d and type(
                q) is Point_d and p.dimension() == q.dimension() and p.dimension() % 2 == 0 and q.dimension() % 2 == 0:
            U, V = Metric_Epsilon_Inf.points_d_to_list(p, q)
            U, V = np.array(U), np.array(V)
            epsilon_inf = Metric_Epsilon_Inf.congruence_objective_l_inf(U, V)
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
            U, V = Metric_Epsilon_Inf.points_to_list(p, q)
            U, V = np.array(U), np.array(V)
            epsilon_inf = Metric_Epsilon_Inf.congruence_objective_l_inf(U, V)
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
