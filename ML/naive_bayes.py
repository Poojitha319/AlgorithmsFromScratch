import numpy as np
def softmax(z):
        e = np.exp(z - np.amax(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
class BaseEstimator:
    y_required = True
    fit_required = True

    def _setup_input(self, X, y=None):
        """Ensure inputs to an estimator are in the expected format.

        Ensures X and y are stored as numpy ndarrays by converting from an
        array-like object if necessary. Enables estimators to define whether
        they require a set of y target values or not with y_required, e.g.
        kmeans clustering requires no target labels and is fit against only X.

        Parameters
        ----------
        X : array-like
            Feature dataset.
        y : array-like
            Target values. By default is required, but if y_required = false
            then may be omitted.
        """
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

class NaiveBayesClassifier(BaseEstimator):
    """Gaussian Naive Bayes."""

    # Binary problem.
    n_classes = 2

    def fit(self, X, y=None):
        self._setup_input(X, y)
        # Check target labels
        assert list(np.unique(y)) == [0, 1]

        # Mean and variance for each class and feature combination
        self._mean = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self._var = np.zeros((self.n_classes, self.n_features), dtype=np.float64)

        self._priors = np.zeros(self.n_classes, dtype=np.float64)

        for c in range(self.n_classes):
            # Filter features by class
            X_c = X[y == c]

            # Calculate mean, variance, prior for each class
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(X.shape[0])
    
    def _predict(self, X=None):
        # Apply _predict_proba for each row
        predictions = np.apply_along_axis(self._predict_row, 1, X)

        # Normalize probabilities so that each row will sum up to 1.0
        return softmax(predictions)

    def _predict_row(self, x):
        """Predict log likelihood for given row."""
        output = []
        for y in range(self.n_classes):
            prior = np.log(self._priors[y])
            posterior = np.log(self._pdf(y, x)).sum()
            prediction = prior + posterior

            output.append(prediction)
        return output

    def _pdf(self, n_class, x):
        """Calculate Gaussian PDF for each feature."""

        mean = self._mean[n_class]
        var = self._var[n_class]

        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
