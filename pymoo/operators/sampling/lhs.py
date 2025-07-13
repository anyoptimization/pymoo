import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.util import default_random_state
from pymoo.util.misc import cdist


def criterion_maxmin(X):
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    return np.min(D)


def criterion_corr(X):
    M = np.corrcoef(X.T, rowvar=True)
    return -np.sum(np.tril(M, -1) ** 2)


@default_random_state
def sampling_lhs(n_samples, n_var, xl=0, xu=1, smooth=True, criterion=criterion_maxmin, n_iter=50, random_state=None):

    X = sampling_lhs_unit(n_samples, n_var, smooth=smooth, random_state=random_state)

    # if a criterion is selected to further improve the sampling
    if criterion is not None:

        # current best score is stored here
        score = criterion(X)

        for j in range(1, n_iter):

            # create new random sample and check the score again
            _X = sampling_lhs_unit(n_samples, n_var, smooth=smooth, random_state=random_state)
            _score = criterion(_X)

            if _score > score:
                X, score = _X, _score

    return xl + X * (xu - xl)


@default_random_state
def sampling_lhs_unit(n_samples, n_var, smooth=True, random_state=None):
    X = random_state.random(size=(n_samples, n_var))
    Xp = X.argsort(axis=0) + 1

    if smooth:
        Xp = Xp - random_state.random(Xp.shape)
    else:
        Xp = Xp - 0.5
    Xp /= n_samples
    return Xp


class LatinHypercubeSampling(Sampling):

    def __init__(self,
                 smooth=True,
                 iterations=20,
                 criterion=criterion_maxmin) -> None:
        super().__init__()
        self.smooth = smooth
        self.iterations = iterations
        self.criterion = criterion

    def _do(self, problem, n_samples, random_state=None, **kwargs):
        xl, xu = problem.bounds()

        X = sampling_lhs(n_samples, problem.n_var, xl=xl, xu=xu, smooth=self.smooth,
                         criterion=self.criterion, n_iter=self.iterations, random_state=random_state)

        return X


class LHS(LatinHypercubeSampling):
    pass
