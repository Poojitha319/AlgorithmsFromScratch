import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from .kmeans import KMeans

class BaseEstimator:
    y_required = True
    fit_required = True

    def _setup_input(self, features, targets=None):
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        if features.size == 0:
            raise ValueError("Got an empty matrix.")

        self.n_samples, self.n_features = (1, features.shape) if features.ndim == 1 else features.shape

        self.features = features

        if self.y_required:
            if targets is None:
                raise ValueError("Missed required argument y")

            if not isinstance(targets, np.ndarray):
                targets = np.array(targets)

            if targets.size == 0:
                raise ValueError("The targets array must be non-empty.")

        self.targets = targets

    def fit(self, features, targets=None):
        self._setup_input(features, targets)

    def predict(self, features=None):
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        if self.features is not None or not self.fit_required:
            return self._predict(features)
        else:
            raise ValueError("You must call `fit` before `predict`")

    def _predict(self, features=None):
        raise NotImplementedError()


class GaussianMixture(BaseEstimator):
    y_required = False

    def __init__(self, num_components=4, initialization="random", max_iterations=500, tolerance=1e-3):
        self.num_components = num_components
        self.max_iterations = max_iterations
        self.initialization = initialization
        self.assignments = None
        self.likelihoods = []
        self.tolerance = tolerance

    def fit(self, features, targets=None):
        """Perform Expectation–Maximization (EM) until converged."""
        self._setup_input(features, targets)
        self._initialize()
        for _ in range(self.max_iterations):
            self._E_step()
            self._M_step()
            if self._is_converged():
                break

    def _initialize(self):
        self.weights = np.ones(self.num_components)
        if self.initialization == "random":
            self.means = [self.features[x] for x in random.sample(range(self.n_samples), self.num_components)]
            self.covariances = [np.cov(self.features.T) for _ in range(self.num_components)]

        elif self.initialization == "kmeans":
            kmeans = KMeans(K=self.num_components, max_iters=self.max_iterations // 3, init="++")
            kmeans.fit(self.features)
            self.assignments = kmeans.predict()
            self.means = kmeans.centroids
            self.covariances = []
            for i in np.unique(self.assignments):
                self.weights[int(i)] = (self.assignments == i).sum()
                self.covariances.append(np.cov(self.features[self.assignments == i].T))
        else:
            raise ValueError("Unknown type of initialization parameter")
        self.weights /= self.weights.sum()

    def _E_step(self):
        """Expectation (E-step) for Gaussian Mixture."""
        likelihoods = self._get_likelihood(self.features)
        self.likelihoods.append(likelihoods.sum())
        weighted_likelihoods = self._get_weighted_likelihood(likelihoods)
        self.assignments = weighted_likelihoods.argmax(axis=1)
        weighted_likelihoods /= weighted_likelihoods.sum(axis=1, keepdims=True)
        self.responsibilities = weighted_likelihoods

    def _M_step(self):
        """Maximization (M-step) for Gaussian Mixture."""
        weights_sum = self.responsibilities.sum(axis=0)
        for assignment in range(self.num_components):
            resp = self.responsibilities[:, assignment][:, np.newaxis]
            self.means[assignment] = (resp * self.features).sum(axis=0) / resp.sum()
            self.covariances[assignment] = (self.features - self.means[assignment]).T.dot(
                (self.features - self.means[assignment]) * resp
            ) / weights_sum[assignment]
        self.weights = weights_sum / weights_sum.sum()

    def _is_converged(self):
        """Check if the difference of the latest two likelihoods is less than the tolerance."""
        if len(self.likelihoods) > 1 and (self.likelihoods[-1] - self.likelihoods[-2] <= self.tolerance):
            return True
        return False

    def _predict(self, features):
        """Get the assignments for features with GMM clusters."""
        if not features.shape:
            return self.assignments
        likelihoods = self._get_likelihood(features)
        weighted_likelihoods = self._get_weighted_likelihood(likelihoods)
        assignments = weighted_likelihoods.argmax(axis=1)
        return assignments

    def _get_likelihood(self, data):
        num_data = data.shape[0]
        likelihoods = np.zeros((num_data, self.num_components))
        for component in range(self.num_components):
            likelihoods[:, component] = multivariate_normal.pdf(data, self.means[component], self.covariances[component])
        return likelihoods

    def _get_weighted_likelihood(self, likelihood):
        return self.weights * likelihood

    def plot(self, data=None, ax=None, hold_on=False):
        """Plot contour for 2D data."""
        if not (len(self.features.shape) == 2 and self.features.shape[1] == 2):
            raise AttributeError("Only support for visualizing 2D data.")

        if ax is None:
            _, ax = plt.subplots()

        if data is None:
            data = self.features
            assignments = self.assignments
        else:
            assignments = self.predict(data)

        colors = "bgrcmyk"
        cmap = lambda assignment: colors[int(assignment) % len(colors)]

        # Generate grid
        delta = 0.025
        margin = 0.2
        xmax, ymax = self.features.max(axis=0) + margin
        xmin, ymin = self.features.min(axis=0) - margin
        axis_X, axis_Y = np.meshgrid(np.arange(xmin, xmax, delta), np.arange(ymin, ymax, delta))

        def grid_gaussian_pdf(mean, cov):
            grid_array = np.column_stack([axis_X.flatten(), axis_Y.flatten()])
            return multivariate_normal.pdf(grid_array, mean, cov).reshape(axis_X.shape)

        # Plot scatters
        scatter_colors = [cmap(assignment) for assignment in assignments] if assignments is not None else None
        ax.scatter(data[:, 0], data[:, 1], c=scatter_colors)

        # Plot contours
        for assignment in range(self.num_components):
            ax.contour(
                axis_X,
                axis_Y,
                grid_gaussian_pdf(self.means[assignment], self.covariances[assignment]),
                colors=cmap(assignment),
            )

        if not hold_on:
            plt.show()
