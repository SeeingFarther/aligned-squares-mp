import numpy as np
import sklearn.neighbors

from discopygal.solvers.metrics import *
from discopygal.solvers.nearest_neighbors import NearestNeighbors


class NearestNeighbors_sklearn_ball(NearestNeighbors):
    """
    Sklearn implementation of nearest neighbors

    :param metric: metric to compute nearest neighbors
    :type metric: :class:`~discopygal.solvers.metrics.Metric`
    """

    def __init__(self, metric=Metric_Euclidean):
        super().__init__(metric)
        self.balltree = None
        self.np_points = []
        self.points = []

    def fit(self, points):
        """
        Get a list of points (in CGAL Point_2 or Point_d format) and fit some kind of data structure on them.

        :param points: list of points
        :type points: list<:class:`~discopygal.bindings.Point_2`> or list<:class:`~discopygal.bindings.Point_d`>
        """
        if len(points) == 0:
            return

        # Convert points to numpy array
        self.points = points
        if type(self.points[0]) is Point_2:
            self.np_points = np.zeros((len(self.points), 2))
            for i, point in enumerate(self.points):
                self.np_points[i, 0] = point.x().to_double()
                self.np_points[i, 1] = point.y().to_double()
        elif type(self.points[0]) is Point_d:
            d = self.points[0].dimension()
            self.np_points = np.zeros((len(self.points), d))
            for i, point in enumerate(self.points):
                for j in range(d):
                    self.np_points[i, j] = point[j]
        else:
            raise Exception("points should be either Point_2 or Point_d")
        self.balltree = sklearn.neighbors.BallTree(self.np_points, metric=self.metric.sklearn_impl())

    def k_nearest(self, point, k):
        """
        Given a point, return the k-nearest neighbors to that point.


        :param point: query point
        :type point: :class:`~discopygal.bindings.Point_2` or :class:`~discopygal.bindings.Point_d`
        :param k: number of neighbors to return (k)
        :type k: int

        :return: k nearest neighbors
        :rtype: list<:class:`~discopygal.bindings.Point_2`> or list<:class:`~discopygal.bindings.Point_d`>
        """
        if self.balltree is None:
            return []
        if type(point) is Point_2:
            np_point = np.array([point.x().to_double(), point.y().to_double()]).reshape((-1, 2))
        elif type(point) is Point_d:
            d = point.dimension()
            np_point = np.zeros((1, d))
            for i in range(d):
                np_point[0, i] = point[i]
        else:
            raise Exception("points should be either Point_2 or Point_d")
        _, indices = self.balltree.query(np_point, k=k)
        res = []
        for idx in indices[0]:
            res.append(self.points[idx])
        return res

    def nearest_in_radius(self, point, radius):
        """
        Given a point and a radius, return all the neighbors that have distance <= radius

        :param point: query point
        :type point: :class:`~discopygal.bindings.Point_2`
        :param radius: radius of neighborhood
        :type radius: :class:`~discopygal.bindings.FT`

        :return: nearest neighbors in radius
        :rtype: list<:class:`~discopygal.bindings.Point_2`>
        """
        if self.balltree is None:
            return []
        if type(point) is Point_2:
            np_point = np.array([point.x().to_double(), point.y().to_double()]).reshape((-1, 2))
        elif type(point) is Point_d:
            d = point.dimension()
            np_point = np.zeros((1, d))
            for i in range(d):
                np_point[0, i] = point[i].to_double()
        else:
            raise Exception("points should be either Point_2 or Point_d")
        indices = self.balltree.query_radius(np_point, r=radius.to_double())
        res = []
        for idx in indices[0]:
            res.append(self.points[idx])
        return res