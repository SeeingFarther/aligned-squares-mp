import sklearn.metrics
import numpy as np
from discopygal.bindings import *
from discopygal.solvers.metrics import Metric


class MetricNotImplemented(Exception):
    pass


class Metric_CTD(Metric):
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
            r = p - q
            b = np.power(r.y().to_double(), 2)
            b += np.power(r.x().to_double(), 2)
            a = b
            b /= 2
            d = a - b
            return FT(d)
        elif type(p) is Point_d and type(
                q) is Point_d and p.dimension() == q.dimension() and p.dimension() % 2 == 0 and q.dimension() % 2 == 0:
            r = p - q
            a, sum_x, sum_y = 0, 0, 0
            length = r.dimension() / 2
            for i in range(length):
                x = r[2 * i]
                y = r[2 * i + 1]
                sum_x += x
                sum_y += y
                a += np.power(x, 2) + np.power(y, 2)
            b = np.power(sum_x, 2) + np.power(sum_y, 2)
            b /= length
            d = a - b
            return FT(d)
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
            r = p - q
            a, sum_x, sum_y = 0, 0, 0
            length = int(len(r) / 2)
            for i in range(length):
                x = r[2 * i]
                y = r[2 * i + 1]
                sum_x += x
                sum_y += y
                a += np.power(x, 2) + np.power(y, 2)
            b = np.power(sum_x, 2) + np.power(sum_y, 2)
            b /= length
            d = a - b
            return d
        else:
            raise MetricNotImplemented('p,q should be same dimension or Not even dimension')

    @staticmethod
    def sklearn_impl():
        """
        Return the metric as sklearn metric object.
        """
        # Implementation specific to scikit-learn
        return sklearn.metrics.DistanceMetric.get_metric(metric='pyfunc', func=Metric_CTD.float_dist)
