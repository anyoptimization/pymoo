import numpy as np

from pymoo.model.sampling import Sampling
from pymoo.util.misc import cdist


class LatinHypercubeSampling(Sampling):
    """
    Latin Hypercube Sampling

    Implementation is similar to the Matlab lhsdesign method and offers the same options for the sampling.
    """

    def __init__(self,
                 smooth=True,
                 iterations=20,
                 criterion="maxmin") -> None:
        super().__init__()
        self.smooth = smooth
        self.iterations = iterations
        self.criterion = criterion

    def _sample(self, n_samples, n_var):
        X = np.random.random(size=(n_samples, n_var))
        val = X.argsort(axis=0) + 1

        if self.smooth:
            val = val - np.random.random(val.shape)
        else:
            val = val - 0.5
        val /= n_samples

        return val

    def _calc_score(self, X):

        if isinstance(self.criterion, str):

            if self.criterion == "maxmin":
                D = cdist(X, X)
                np.fill_diagonal(D, np.inf)
                return np.min(D)

            elif self.criterion == "correlation":
                M = np.corrcoef(X.T, rowvar=True)
                return -np.sum(np.tril(M, -1) ** 2)

            else:
                raise Exception("Unknown criterion.")
        elif callable(self.criterion):
            return self.criterion(X)

        else:
            raise Exception("Either provide a str or a function as a criterion!")

    def _do(self, problem, n_samples, **kwargs):

        # sample for the first time -
        X = self._sample(n_samples, problem.n_var)

        # if a criterion is selected to further improve the sampling
        if self.criterion is not None:

            # current best score is stored here
            score = self._calc_score(X)

            for j in range(self.iterations - 1):

                # create new random sample and check the score again
                _X = self._sample(n_samples, problem.n_var)
                _score = self._calc_score(_X)

                if _score > score:
                    X, score = _X, _score

        for i in range(problem.n_var):
            X[:, i] = X[:, i] * (problem.xu[i] - problem.xl[i]) + problem.xl[i]

        return X
