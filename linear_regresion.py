import numpy as np

class LinearRegression:
    
    def __init__(self, learning_rate=0.001, n_iters_gd=1000):
        self.learning_rate = learning_rate
        self.n_iters_gd = n_iters_gd
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initializing weights and bias
        # random values also could be used for weight and bias assignment
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descendent to update weight and bias using 
        for _ in range(self.n_iters_gd):
            # multiplies feature vector with weights and adds biases
            y_predicted = np.dot(X, self.weights) + self.bias
            # differentiating cost function (MSE) with b, scaling factor (i.e 2) and negative sign are omitted
            derivate_w = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            # differentiating cost function (MSE) with b, scaling factor (i.e 2) and negative sign are omitted
            derivate_b = (1/n_samples) * np.sum(y_predicted - y)
            # Updating weights and bias
            self.weights -= self.learning_rate * derivate_w
            self.bias -= self.learning_rate * derivate_b

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
