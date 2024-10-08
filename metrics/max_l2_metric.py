import math

import sklearn.metrics

from discopygal.bindings import *
from discopygal.solvers.metrics import Metric


class MetricNotImplemented(Exception):
    pass


class Metric_Max_L2(Metric):
    """
    Implementation of the Euclidean metric for nearest neighbors search
    """

    @staticmethod
    def dist(p, q):
        """
        Return the Max L2 distance between two points for metric

        :param p: first point
        :type p: :class:`~discopygal.bindings.Point_2` or :class:`~discopygal.bindings.Point_d`
        :param q: second point
        :type q: :class:`~discopygal.bindings.Point_2` or :class:`~discopygal.bindings.Point_d`

        :return: distance between p and q
        :rtype: :class:`~discopygal.bindings.FT`
        """
        if type(p) is Point_2 and type(q) is Point_2:
            d = (p.x() - q.x()) * (p.x() - q.x()) + (p.y() - q.y()) * (p.y() - q.y())
            d = math.sqrt(d.to_double())
            return FT(d)
        elif type(p) is Point_d and type(q) is Point_d and p.dimension() == q.dimension():
            d = FT(0)
            for i in range(int(p.dimension() / 2)):
                d += ((p[i] - q[i]) * (p[i] - q[i])) + ((p[i+1] - q[i+1]) * (p[i+1] - q[i+1]))
                d = max(d, math.sqrt(d))
            return FT(d)
        else:
            raise MetricNotImplemented('p,q should be Point_2 or Point_d')


    @staticmethod
    def float_dist(p, q):
        """
        Return the distance between two points consider each 2 coordinates as a point_2 and those
         Max L2 distance for each of these points separately and sum them up Sklearn implementation

        :param p: first point
        :type p: :class:`~discopygal.bindings.Point_2` or :class:`~discopygal.bindings.Point_d`
        :param q: second point
        :type q: :class:`~discopygal.bindings.Point_2` or :class:`~discopygal.bindings.Point_d`

        :return: distance between p and q
        :rtype: :class:`~discopygal.bindings.FT`
        """

        if len(p) == len(q) and len(p) % 2 == 0 and len(q) % 2 == 0:
            d = 0
            for i in range(int(len(p) / 2)):
                dist = 0
                dist += ((p[i] - q[i]) * (p[i] - q[i])) + ((p[i + 1] - q[i + 1]) * (p[i + 1] - q[i + 1]))
                d = max(d, math.sqrt(dist))

            return d
        else:
            raise MetricNotImplemented('p,q should be Point_2 or Point_d')

    @staticmethod
    def sklearn_impl() -> sklearn.metrics.DistanceMetric:
        """
        Return the metric as sklearn metric object.
        """
        # Implementation specific to scikit-learn
        return sklearn.metrics.DistanceMetric.get_metric(metric='pyfunc', func=Metric_Max_L2.float_dist)