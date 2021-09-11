import numpy as np

class SVM:

    def __init__(self,  learning_rate=0.001, lambda_param=0.01, n_iters_gd=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters_gd = n_iters_gd
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters_gd):
            for idx, x_i in enumerate(X):
                condition = y_[idx] - (np.dot(x_i, self.w) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)