from collections import Counter
import numpy as np

def eucledian_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    # initialize with number of nearest neighbours
    def __init__(self, k=3):
        self.k = k
    
    # fits k-NN algorithm
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    # predicts based on fitted model
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute distances between new test sample and all training samples
        distances = [eucledian_distance(x, x_train) for x_train in self.X_train]
        
        # get k nearest points and corresponding lables
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # majority vote, most common class label
        most_common = Counter(k_nearest_labels)
        return most_common[0][0]