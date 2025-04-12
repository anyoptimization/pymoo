import numpy as np


class Optimizer:

    def __init__(self, precision=1e-6) -> None:
        super().__init__()
        self.has_converged = False
        self.precision = precision

    def next(self, X, dX):
        _X = self._next(X, dX)

        if np.abs(_X - X).mean() < self.precision:
            self.has_converged = True

        return _X


class GradientDescent(Optimizer):

    def __init__(self, learning_rate=0.01, **kwargs) -> None:
        super().__init__(**kwargs)
        self.learning_rate = learning_rate

    def _next(self, X, dX):
        return X - self.learning_rate * dX


class Adam(Optimizer):

    def __init__(self, alpha=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, **kwargs) -> None:
        super().__init__(**kwargs)

        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m_t = 0
        self.v_t = 0
        self.t = 0

    def _next(self, X, dX):
        self.t += 1
        beta_1, beta_2 = self.beta_1, self.beta_2

        # update moving average of gradient and squared gradient
        self.m_t = beta_1 * self.m_t + (1 - beta_1) * dX
        self.v_t = beta_2 * self.v_t + (1 - beta_2) * (dX * dX)

        # calculates the bias-corrected estimates
        m_cap = self.m_t / (1 - (beta_1 ** self.t))
        v_cap = self.v_t / (1 - (beta_2 ** self.t))

        # do the gradient update
        _X = X - (self.alpha * m_cap) / (np.sqrt(v_cap) + self.epsilon)

        return _X
