import autograd.numpy as anp
import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
from pymoo.util.misc import stack
from pymoo.model.problem import Problem

import multiprocessing

class MyProblem(Problem):
    def __init__(self, **kwargs):
        super().__init__(n_var = 10, n_obj = 1, n_constr = 0, xl = -5, xu = 5,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (x ** 2).sum(axis = -1)

with multiprocessing.Pool() as pool:
    problem = MyProblem(elementwise_evaluation = True, parallelization = ('starmap', pool.starmap))
    r = np.random.RandomState(seed = 1)
    X = r.random((5, problem.n_var))
    result = problem.evaluate(X).squeeze()
expected = (X ** 2).sum(axis = -1)

print(result)
print(expected)

assert np.all(result == expected)
print('yay')
