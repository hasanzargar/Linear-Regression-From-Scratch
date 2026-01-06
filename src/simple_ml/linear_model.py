import numpy as np


class LinearRegression:
    def __init__(self, learning_rate, n_iters):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.asarray(X)
        n_samples, n_features = np.shape(X)
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predict = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(2 * (X.T), (y_predict - y))
            db = (1 / n_samples) * np.sum(2 * (y_predict - y))
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, x):
        y_predict = np.dot(x, self.weights) + self.bias
        return y_predict
