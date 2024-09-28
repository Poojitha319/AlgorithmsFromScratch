import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .metrics.distance import euclidean_distance

random.seed(1111)

class BaseEstimator:
    y_required = True
    fit_required = True

    def _setup_input(self, X, y=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError("Got an empty matrix.")

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError("Missed required argument y")

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError("The targets array must be no-empty.")

        self.y = y

    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError("You must call `fit` before `predict`")

    def _predict(self, X=None):
        raise NotImplementedError()
    
class KMeans(BaseEstimator):
    

    y_required = False

    def __init__(self, K=5, max_iters=100, init="random"):
        self.K = K
        self.max_iters = max_iters
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
        self.init = init

    def _initialize_centroids(self, init):
        """Set the initial centroids."""

        if init == "random":
            self.centroids = [self.X[x] for x in random.sample(range(self.n_samples), self.K)]
        elif init == "++":
            self.centroids = [random.choice(self.X)]
            while len(self.centroids) < self.K:
                self.centroids.append(self._choose_next_center())
        else:
            raise ValueError("Unknown type of init parameter")

    def _predict(self, X=None):
        """Perform clustering on the dataset."""
        self._initialize_centroids(self.init)
        centroids = self.centroids

        # Optimize clusters
        for _ in range(self.max_iters):
            self._assign(centroids)
            centroids_old = centroids
            centroids = [self._get_centroid(cluster) for cluster in self.clusters]

            if self._is_converged(centroids_old, centroids):
                break

        self.centroids = centroids

        return self._get_predictions()

    def _get_predictions(self):
        predictions = np.empty(self.n_samples)

        for i, cluster in enumerate(self.clusters):
            for index in cluster:
                predictions[index] = i
        return predictions

    def _assign(self, centroids):

        for row in range(self.n_samples):
            for i, cluster in enumerate(self.clusters):
                if row in cluster:
                    self.clusters[i].remove(row)
                    break

            closest = self._closest(row, centroids)
            self.clusters[closest].append(row)

    def _closest(self, fpoint, centroids):
        """Find the closest centroid for a point."""
        closest_index = None
        closest_distance = None
        for i, point in enumerate(centroids):
            dist = euclidean_distance(self.X[fpoint], point)
            if closest_index is None or dist < closest_distance:
                closest_index = i
                closest_distance = dist
        return closest_index

    def _get_centroid(self, cluster):
        """Get values by indices and take the mean."""
        return [np.mean(np.take(self.X[:, i], cluster)) for i in range(self.n_features)]

    def _dist_from_centers(self):
        """Calculate distance from centers."""
        return np.array([min([euclidean_distance(x, c) for c in self.centroids]) for x in self.X])

    def _choose_next_center(self):
        distances = self._dist_from_centers()
        squared_distances = distances ** 2
        probs = squared_distances / squared_distances.sum()
        ind = np.random.choice(self.X.shape[0], 1, p=probs)[0]
        return self.X[ind]

    def _is_converged(self, centroids_old, centroids):
        """Check if the distance between old and new centroids is zero."""
        distance = 0
        for i in range(self.K):
            distance += euclidean_distance(centroids_old[i], centroids[i])
        return distance == 0

    def plot(self, ax=None, holdon=False):
        sns.set(style="white")
        palette = sns.color_palette("hls", self.K + 1)
        data = self.X

        if ax is None:
            _, ax = plt.subplots()

        for i, index in enumerate(self.clusters):
            point = np.array(data[index]).T
            ax.scatter(*point, c=[palette[i], ])

        for point in self.centroids:
            ax.scatter(*point, marker="x", linewidths=10)

        if not holdon:
            plt.show()
