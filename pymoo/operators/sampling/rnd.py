"""Random sampling operators."""

import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.util import default_random_state


@default_random_state
def random(problem, n_samples: int = 1, random_state=None) -> np.ndarray:
    """Generate random samples.

    Args:
        problem: Optimization problem.
        n_samples: Number of samples to generate.
        random_state: Random state for reproducibility.

    Returns:
        Sample matrix of shape (n_samples, n_var).
    """
    X = random_state.random((n_samples, problem.n_var))

    if problem.has_bounds():
        xl, xu = problem.bounds()
        assert np.all(xu >= xl)
        X = xl + (xu - xl) * X

    return X


class FloatRandomSampling(Sampling):
    """Random sampling operator for continuous variables."""

    def _do(  # type: ignore[override]
        self, problem, n_samples: int, *args, random_state=None, **kwargs
    ) -> np.ndarray:
        """Generate random float samples within problem bounds.

        Args:
            problem: Optimization problem.
            n_samples: Number of samples.
            *args: Additional positional arguments.
            random_state: Random state for reproducibility.
            **kwargs: Additional keyword arguments.

        Returns:
            Sample matrix of shape (n_samples, n_var).
        """
        X = random_state.random((n_samples, problem.n_var))

        if problem.has_bounds():
            xl, xu = problem.bounds()
            assert np.all(xu >= xl)
            X = xl + (xu - xl) * X

        return X


class BinaryRandomSampling(Sampling):
    """Random sampling operator for binary variables."""

    def _do(  # type: ignore[override]
        self, problem, n_samples: int, *args, random_state=None, **kwargs
    ) -> np.ndarray:
        """Generate random binary samples.

        Args:
            problem: Optimization problem.
            n_samples: Number of samples.
            *args: Additional positional arguments.
            random_state: Random state for reproducibility.
            **kwargs: Additional keyword arguments.

        Returns:
            Sample matrix of shape (n_samples, n_var) with boolean values.
        """
        val = random_state.random((n_samples, problem.n_var))
        return (val < 0.5).astype(bool)


class IntegerRandomSampling(FloatRandomSampling):
    """Random sampling operator for integer variables."""

    def _do(  # type: ignore[override]
        self, problem, n_samples: int, *args, random_state=None, **kwargs
    ) -> np.ndarray:
        """Generate random integer samples within problem bounds.

        Args:
            problem: Optimization problem.
            n_samples: Number of samples.
            *args: Additional positional arguments.
            random_state: Random state for reproducibility.
            **kwargs: Additional keyword arguments.

        Returns:
            Sample matrix of shape (n_samples, n_var) with integer values.
        """
        n, (xl, xu) = problem.n_var, problem.bounds()
        return np.column_stack([random_state.integers(xl[k], xu[k] + 1, size=n_samples) for k in range(n)])


class PermutationRandomSampling(Sampling):
    """Random sampling operator for permutation variables."""

    def _do(  # type: ignore[override]
        self, problem, n_samples: int, *args, random_state=None, **kwargs
    ) -> np.ndarray:
        """Generate random permutation samples.

        Args:
            problem: Optimization problem.
            n_samples: Number of samples.
            *args: Additional positional arguments.
            random_state: Random state for reproducibility.
            **kwargs: Additional keyword arguments.

        Returns:
            Sample matrix of shape (n_samples, n_var) with permutation values.
        """
        X = np.full((n_samples, problem.n_var), 0, dtype=int)
        for i in range(n_samples):
            X[i, :] = random_state.permutation(problem.n_var)
        return X
