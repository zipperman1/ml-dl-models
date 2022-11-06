import numpy as np


class LinearRegression:
    def __init__(self, **kwargs):
        self.coef_ = None

    def fit(self, x: np.array, y: np.array):
        x = np.concatenate((x, np.ones(x.shape[0]).reshape(-1, 1)), axis=1)
        self.coef_ = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)
        print(self.coef_)

    def predict(self, x: np.array):
        x = np.concatenate((x, np.ones(x.shape[0]).reshape(-1, 1)), axis=1)
        return np.matmul(x, self.coef_)
