# coding:utf-8

import logging

import numpy as np
from autograd import grad
from .metrics.metrics import mean_squared_error, binary_crossentropy


np.random.seed(1000)

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
            raise ValueError("You must  `fit` the values before `predict`")

    def _predict(self, X=None):
        raise NotImplementedError()


class BasicRegression(BaseEstimator):
    def __init__(self, lr=0.001, penalty="None", C=0.01, tolerance=0.0001, max_iters=1000):
        
        self.C = C
        self.penalty = penalty
        self.tolerance = tolerance
        self.lr = lr
        self.max_iters = max_iters
        self.errors = []
        self.theta = []
        self.n_samples, self.n_features = None, None
        self.cost_func = None

    def _loss(self, w):
        raise NotImplementedError()

    def init_cost(self):
        raise NotImplementedError()

    def _add_penalty(self, loss, w):
        """Apply regularization to the loss."""
        if self.penalty == "l1":
            loss += self.C * np.abs(w[1:]).sum()
        elif self.penalty == "l2":
            loss += (0.5 * self.C) * (w[1:] ** 2).sum()
        return loss

    def _cost(self, X, y, theta):
        prediction = X.dot(theta)
        error = self.cost_func(y, prediction)
        return error

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.init_cost()
        self.n_samples, self.n_features = X.shape

        # Initialize weights + bias term
        self.theta = np.random.normal(size=(self.n_features + 1), scale=0.5)

        # Add an intercept column
        self.X = self._add_intercept(self.X)

        self._train()

    @staticmethod
    def _add_intercept(X):
        b = np.ones([X.shape[0], 1])
        return np.concatenate([b, X], axis=1)

    def _train(self):
        self.theta, self.errors = self._gradient_descent()
        logging.info(" Theta: %s" % self.theta.flatten())

    def _predict(self, X=None):
        X = self._add_intercept(X)
        return X.dot(self.theta)

    def _gradient_descent(self):
        theta = self.theta
        errors = [self._cost(self.X, self.y, theta)]
        # Get derivative of the loss function
        cost_d = grad(self._loss)
        for i in range(1, self.max_iters + 1):
            # Calculate gradient and update theta
            delta = cost_d(theta)
            theta -= self.lr * delta

            errors.append(self._cost(self.X, self.y, theta))
            logging.info("Iteration %s, error %s" % (i, errors[i]))

            error_diff = np.linalg.norm(errors[i - 1] - errors[i])
            if error_diff < self.tolerance:
                logging.info("Convergence has reached.")
                break
        return theta, errors


class LinearRegression(BasicRegression):
    """Linear regression with gradient descent optimizer."""

    def _loss(self, w):
        loss = self.cost_func(self.y, np.dot(self.X, w))
        return self._add_penalty(loss, w)

    def init_cost(self):
        self.cost_func = mean_squared_error


class LogisticRegression(BasicRegression):
    """Binary logistic regression with gradient descent optimizer."""

    def init_cost(self):
        self.cost_func = binary_crossentropy

    def _loss(self, w):
        loss = self.cost_func(self.y, self.sigmoid(np.dot(self.X, w)))
        return self._add_penalty(loss, w)

    @staticmethod
    def sigmoid(x):
        return 0.5 * (np.tanh(0.5 * x) + 1)

    def _predict(self, X=None):
        X = self._add_intercept(X)
        return self.sigmoid(X.dot(self.theta))
