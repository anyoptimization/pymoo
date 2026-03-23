import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.util import default_random_state


@default_random_state
def random(problem, n_samples=1, random_state=None):
    X = random_state.random((n_samples, problem.n_var))

    if problem.has_bounds():
        xl, xu = problem.bounds()
        assert np.all(xu >= xl)
        X = xl + (xu - xl) * X

    return X


class FloatRandomSampling(Sampling):

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        X = random_state.random((n_samples, problem.n_var))

        if problem.has_bounds():
            xl, xu = problem.bounds()
            assert np.all(xu >= xl)
            X = xl + (xu - xl) * X

        return X


class BinaryRandomSampling(Sampling):

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        val = random_state.random((n_samples, problem.n_var))
        return (val < 0.5).astype(bool)


class IntegerRandomSampling(FloatRandomSampling):
    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        xl, xu = problem.bounds()
        return random_state.integers(xl, xu + 1, size=(n_samples, problem.n_var))


class PermutationRandomSampling(Sampling):
    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        return np.argsort(random_state.random((n_samples, problem.n_var)), axis=1)
