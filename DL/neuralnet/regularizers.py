
import numpy as np
from autograd import elementwise_grad
class Regularizer(object):
    def __init__(self, C=0.01):
        self.C = C
        self._grad = elementwise_grad(self._penalty)

    def _penalty(self, weights):
        raise NotImplementedError()

    def grad(self, weights):
        return self._grad(weights)

    def __call__(self, weights):
        return self.grad(weights)


class L1(Regularizer):
    def _penalty(self, weights):
        return self.C * np.abs(weights)


class L2(Regularizer):
    def _penalty(self, weights):
        return self.C * weights ** 2

class Dropout:
    def __init__(self, rate=0.5):
        self.rate = rate

    def __call__(self, weights):
        if self.rate > 0:
            mask = np.random.binomial(1, 1 - self.rate, size=weights.shape)
            return weights * mask / (1 - self.rate)  # scale the weights
        return weights
    

class ActivityRegularization(Regularizer):
    """Applies a penalty based on the output of the layer."""
    def _penalty(self, output):
        return self.C * np.sum(np.abs(output))  # L1 activity regularization
    
class WeightNoise(Regularizer):
    """Adds noise to weights during training."""
    def __init__(self, stddev=0.01):
        self.stddev = stddev

    def __call__(self, weights):
        noise = np.random.normal(0, self.stddev, size=weights.shape)
        return weights + noise
    
class SpectralRegularization(Regularizer):
    """Regularizes based on the spectral norm of weights."""
    def _penalty(self, weights):
        u, s, vh = np.linalg.svd(weights, full_matrices=False)
        return self.C * np.max(s)


class ElasticNet(Regularizer):
    """Linear combination of L1 and L2 penalties."""

    def _penalty(self, weights):
        return 0.5 * self.C * weights ** 2 + (1.0 - self.C) * np.abs(weights)
