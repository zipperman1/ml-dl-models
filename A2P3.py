import numpy as np


class UnfittedError(Exception):
    pass


class KNN_classifier:
    def __init__(self, n_neighbors: int, **kwargs):
        self.K = n_neighbors
        self.params = None
        self.markers = None

    def fit(self, x: np.array, y: np.array):
        self.params = x
        self.markers = y

    def predict(self, x: np.array):
        predictions = []

        if self.params is None:
            return 'UnfittedError'

        for point in x:
            point_dist = np.linalg.norm(self.params - point, axis=1)
            neighbors = self.markers[np.argsort(point_dist)]
            cls, count = np.unique(neighbors[:self.K], return_counts=True, axis=0)
            cls_count = dict(zip(cls, count))

            predictions += [max(cls_count, key=cls_count.get)]

        return predictions