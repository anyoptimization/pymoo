import numpy as np

from pymoo.algorithms.so_gradient_descent import GradientBasedAlgorithm


class Adam(GradientBasedAlgorithm):

    def __init__(self,
                 X,
                 alpha=0.005,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 **kwargs) -> None:
        super().__init__(X, **kwargs)

        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.t = 0
        self.m_t = 0
        self.v_t = 0

    def apply(self):
        X, dX = self.X, self.dX

        self.t += 1
        beta_1, beta_2 = self.beta_1, self.beta_2

        # update moving average of gradient and squared gradient
        self.m_t = beta_1 * self.m_t + (1 - beta_1) * dX
        self.v_t = beta_2 * self.v_t + (1 - beta_2) * (dX * dX)

        # calculates the bias-corrected estimates
        m_cap = self.m_t / (1 - (beta_1 ** self.t))
        v_cap = self.v_t / (1 - (beta_2 ** self.t))

        # do the gradient update
        self.X = X - (self.alpha * m_cap) / (np.sqrt(v_cap) + self.epsilon)

    def restart(self):
        self.t = 0
        self.m_t = 0
        self.v_t = 0
        self.alpha /= 2


