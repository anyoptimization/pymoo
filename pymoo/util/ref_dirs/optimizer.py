"""Gradient-based optimization algorithms for reference direction computation."""

from numpy.typing import NDArray
import numpy as np


class Optimizer:
    """Base optimizer for iterative updates with convergence detection."""

    def __init__(self, precision: float = 1e-6) -> None:
        """Initialize the optimizer.

        Args:
            precision: Convergence threshold for parameter changes.
        """
        super().__init__()
        self.has_converged = False
        self.precision = precision

    def next(self, X: NDArray, dX: NDArray) -> NDArray:
        """Perform one optimization step.

        Args:
            X: Current parameters.
            dX: Parameter gradients.

        Returns:
            Updated parameters.
        """
        _X = self._next(X, dX)

        if np.abs(_X - X).mean() < self.precision:
            self.has_converged = True

        return _X

    def _next(self, X: NDArray, dX: NDArray) -> NDArray:  # pragma: no cover
        """Subclass implementation for parameter update.

        Args:
            X: Current parameters.
            dX: Parameter gradients.

        Returns:
            Updated parameters.
        """
        raise NotImplementedError


class GradientDescent(Optimizer):
    """Gradient descent optimizer."""

    def __init__(self, learning_rate: float = 0.01, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Initialize gradient descent optimizer.

        Args:
            learning_rate: Step size for gradient updates.
            **kwargs: Additional arguments for parent Optimizer.
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate

    def _next(self, X: NDArray, dX: NDArray) -> NDArray:
        """Update parameters using gradient descent.

        Args:
            X: Current parameters.
            dX: Parameter gradients.

        Returns:
            Updated parameters.
        """
        return X - self.learning_rate * dX


class Adam(Optimizer):
    """Adam optimizer with adaptive learning rates."""

    def __init__(
        self,
        alpha: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        **kwargs,  # type: ignore[no-untyped-def]
    ) -> None:
        """Initialize Adam optimizer.

        Args:
            alpha: Learning rate.
            beta_1: Exponential decay rate for first moment estimates.
            beta_2: Exponential decay rate for second moment estimates.
            epsilon: Small constant for numerical stability.
            **kwargs: Additional arguments for parent Optimizer.
        """
        super().__init__(**kwargs)

        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m_t: float | NDArray = 0
        self.v_t: float | NDArray = 0
        self.t = 0

    def _next(self, X: NDArray, dX: NDArray) -> NDArray:
        """Update parameters using Adam algorithm.

        Args:
            X: Current parameters.
            dX: Parameter gradients.

        Returns:
            Updated parameters.
        """
        self.t += 1
        beta_1, beta_2 = self.beta_1, self.beta_2

        # update moving average of gradient and squared gradient
        self.m_t = beta_1 * self.m_t + (1 - beta_1) * dX
        self.v_t = beta_2 * self.v_t + (1 - beta_2) * (dX * dX)

        # calculates the bias-corrected estimates
        m_cap = self.m_t / (1 - (beta_1**self.t))
        v_cap = self.v_t / (1 - (beta_2**self.t))

        # do the gradient update
        _X = X - (self.alpha * m_cap) / (np.sqrt(v_cap) + self.epsilon)

        return _X
