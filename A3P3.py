import numpy as np


def sigmoid(X):
    return 1 / (1 + np.e ** (-X))

class LogisticRegression(object):
    def __init__(self):
        self.alpha = None

    def fit(self, x_train, y_train, lr, betta, num_epoch):
        self.alpha = np.ones(x_train.shape[1] + 1)
        for epo in range(num_epoch):
            for i, x in enumerate(x_train):
                y_pred = sigmoid(np.dot(np.append(x, 1), self.alpha))
                gradient = np.append(x, 1) * ((1 - y_train[i]) * y_pred - y_train[i] * (1 - y_pred))
                self.alpha = self.alpha - lr * (gradient + self.alpha * betta)

    def predict(self, X):
        preds = sigmoid(np.dot(X, self.alpha[:-1]) + self.alpha[-1])
        preds[preds >= 0.5], preds[preds < 0.5] = 1, 0
        return preds