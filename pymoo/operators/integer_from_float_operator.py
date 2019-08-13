import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.sampling import Sampling


def apply_float_operation(problem, fun):

    # save the original bounds of the problem
    _xl, _xu = problem.xl, problem.xu

    # copy the arrays of the problem and cast them to float
    xl, xu = problem.xl.astype(np.float), problem.xu.astype(np.float)

    # modify the bounds to match the new crossover specifications and set the problem
    problem.xl = xl - (0.5 - 1e-16)
    problem.xu = xu + (0.5 - 1e-16)

    # perform the crossover
    off = fun()

    # now round to nearest integer for all offsprings
    off = np.rint(off).astype(np.int)

    # reset the original bounds of the problem and design space values
    problem.xl = _xl
    problem.xu = _xu

    return off


class IntegerFromFloatCrossover(Crossover):

    def __init__(self, clazz=None, **kwargs):
        if clazz is None:
            raise Exception("Please define the class of the default crossover to use IntegerFromFloatCrossover.")

        self.crossover = clazz(**kwargs)
        super().__init__(self.crossover.n_parents, self.crossover.n_offsprings, prob=self.crossover.prob)

    def _do(self, problem, X, **kwargs):
        def fun():
            return self.crossover._do(problem, X, **kwargs)

        return apply_float_operation(problem, fun)


class IntegerFromFloatMutation(Mutation):

    def __init__(self, clazz=None, **kwargs):
        if clazz is None:
            raise Exception("Please define the class of the default mutation to use IntegerFromFloatMutation.")

        self.mutation = clazz(**kwargs)
        super().__init__()

    def _do(self, problem, X, **kwargs):
        def fun():
            return self.mutation._do(problem, X, **kwargs)

        return apply_float_operation(problem, fun)


class IntegerFromFloatSampling(Sampling):

    def __init__(self, clazz=None, **kwargs):
        if clazz is None:
            raise Exception("Please define the class of the default sampling to use IntegerFromFloatSampling.")

        self.sampling = clazz(**kwargs)
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        def fun():
            return self.sampling._do(problem, n_samples, **kwargs)

        return apply_float_operation(problem, fun)


