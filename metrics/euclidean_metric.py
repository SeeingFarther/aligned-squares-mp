import math

import sklearn.metrics

from discopygal.bindings import *


class MetricNotImplemented(Exception):
    pass


class Metric(object):
    """
    Representation of a metric for nearest neighbor search.
    Should support all kernels/methods for nearest neighbors
    (like CGAL and sklearn).
    """

    def __init__(self):
        pass

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
            return FT(0)
        elif type(p) is Point_d and type(q) is Point_d:
            return FT(0)
        else:
            raise MetricNotImplemented('p,q should be Point_2 or Point_d')

    @staticmethod
    def CGALPY_impl():
        """
        Return the metric as a CGAL metric object (of the spatial search module)
        """
        raise MetricNotImplemented('CGAL')

    @staticmethod
    def sklearn_impl():
        """
        Return the metric as sklearn metric object
        """
        raise MetricNotImplemented('sklearn')


class Metric_Euclidean(Metric):
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
            d = ((p.x() - q.x()) * (p.x() - q.x())) + ((p.y() - q.y()) * (p.y() - q.y()))
            d = math.sqrt(d.to_double())
            return FT(d)
        elif type(p) is Point_d and type(q) is Point_d and p.dimension() == q.dimension():
            d = FT(0)
            for i in range(int(p.dimension() / 2)):
                dist = ((p[i] - q[i]) * (p[i] - q[i])) +((p[i+1] - q[i+1]) * (p[i+1] - q[i+1]))
                d += math.sqrt(dist)
            return FT(d)
        else:
            raise MetricNotImplemented('p,q should be Point_2 or Point_d')


    @staticmethod
    def float_dist(p, q):
        """
        Return the distance between two points consider each 2 coordinates as a point_2 and those
         Euclidean distance for each of these points separately and sum them up

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
                d += math.sqrt(dist)

            return d
        else:
            raise MetricNotImplemented('p,q should be Point_2 or Point_d')

    @staticmethod
    def sklearn_impl() -> sklearn.metrics.DistanceMetric:
        """
        Return the metric as sklearn metric object.
        """
        # Implementation specific to scikit-learn
        return sklearn.metrics.DistanceMetric.get_metric(metric='pyfunc', func=Metric_Euclidean.float_dist)