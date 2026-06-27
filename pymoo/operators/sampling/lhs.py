"""Latin Hypercube Sampling operators."""

import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.util import default_random_state
from pymoo.util.misc import cdist


def criterion_maxmin(X: np.ndarray) -> np.floating:
    """Maxmin criterion for LHS sampling.

    Args:
        X: Sample matrix.

    Returns:
        Minimum distance between samples.
    """
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    return np.min(D)


def criterion_corr(X: np.ndarray) -> np.floating:
    """Correlation criterion for LHS sampling.

    Args:
        X: Sample matrix.

    Returns:
        Negative sum of squared correlations.
    """
    M = np.corrcoef(X.T, rowvar=True)
    return -np.sum(np.tril(M, -1) ** 2)


@default_random_state
def sampling_lhs(
    n_samples: int,
    n_var: int,
    xl: int | float = 0,
    xu: int | float = 1,
    smooth: bool = True,
    criterion=criterion_maxmin,
    n_iter: int = 50,
    random_state=None,
) -> np.ndarray:
    """Generate Latin Hypercube samples.

    Args:
        n_samples: Number of samples.
        n_var: Number of variables.
        xl: Lower bound (scalar or array).
        xu: Upper bound (scalar or array).
        smooth: Whether to use smooth LHS.
        criterion: Criterion function for optimization.
        n_iter: Number of iterations for criterion optimization.
        random_state: Random state for reproducibility.

    Returns:
        Sample matrix of shape (n_samples, n_var).
    """
    X = sampling_lhs_unit(n_samples, n_var, smooth=smooth, random_state=random_state)

    # if a criterion is selected to further improve the sampling
    if criterion is not None:
        # current best score is stored here
        score = criterion(X)

        for j in range(1, n_iter):
            # create new random sample and check the score again
            _X = sampling_lhs_unit(
                n_samples, n_var, smooth=smooth, random_state=random_state
            )
            _score = criterion(_X)

            if _score > score:
                X, score = _X, _score

    return xl + X * (xu - xl)


@default_random_state
def sampling_lhs_unit(
    n_samples: int,
    n_var: int,
    smooth: bool = True,
    random_state=None,
) -> np.ndarray:
    """Generate unit Latin Hypercube samples (on [0,1]).

    Args:
        n_samples: Number of samples.
        n_var: Number of variables.
        smooth: Whether to use smooth LHS.
        random_state: Random state for reproducibility.

    Returns:
        Sample matrix of shape (n_samples, n_var) with values in [0, 1].
    """
    X = random_state.random(size=(n_samples, n_var))
    Xp = X.argsort(axis=0) + 1

    if smooth:
        Xp = Xp - random_state.random(Xp.shape)
    else:
        Xp = Xp - 0.5
    Xp /= n_samples
    return Xp


class LatinHypercubeSampling(Sampling):
    """Latin Hypercube Sampling operator.

    Generates samples using Latin Hypercube Sampling with optional criterion optimization.

    Attributes:
        smooth: Whether to use smooth LHS.
        iterations: Number of iterations for criterion optimization.
        criterion: Criterion function for optimization.
    """

    def __init__(
        self,
        smooth: bool = True,
        iterations: int = 20,
        criterion=criterion_maxmin,
    ) -> None:
        """Initialize the LHS operator.

        Args:
            smooth: Whether to use smooth LHS.
            iterations: Number of iterations for criterion optimization.
            criterion: Criterion function for optimization.
        """
        super().__init__()
        self.smooth = smooth
        self.iterations = iterations
        self.criterion = criterion

    def _do(  # type: ignore[override]
        self, problem, n_samples: int, *args, random_state=None, **kwargs
    ) -> np.ndarray:
        """Generate Latin Hypercube samples.

        Args:
            problem: Optimization problem.
            n_samples: Number of samples.
            *args: Additional positional arguments.
            random_state: Random state for reproducibility.
            **kwargs: Additional keyword arguments.

        Returns:
            Sample matrix of shape (n_samples, n_var).
        """
        xl, xu = problem.bounds()

        X = sampling_lhs(
            n_samples,
            problem.n_var,
            xl=xl,
            xu=xu,
            smooth=self.smooth,
            criterion=self.criterion,
            n_iter=self.iterations,
            random_state=random_state,
        )

        return X


class LHS(LatinHypercubeSampling):
    """Alias for LatinHypercubeSampling operator."""

    pass  # noqa: E701
