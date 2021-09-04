"""
This class provide an interface for other libraries to specific modules. For example, the evolutionary operations
can be used easily just by calling a function and providing the lower and upper bounds of the problem.

"""

import numpy as np

from pymoo.core.population import Population
from pymoo.core.problem import Problem


# =========================================================================================================
# A global interface for some features
# =========================================================================================================


def get_problem_func(n_var, xl, xu, type_var):
    class P(Problem):
        def __init__(self) -> None:
            super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=xl, xu=xu, type_var=type_var)

    return P


def sample(sampling, n_samples, n_var, xl=0, xu=1, **kwargs):
    problem = get_problem_func(n_var, xl, xu, None)(**kwargs)
    return sampling.do(problem, n_samples, pop=None, **kwargs)


def crossover(crossover, a, b, c=None, xl=0, xu=1, type_var=np.double, **kwargs):
    n = a.shape[0]
    _pop = Population.merge(Population.new("X", a), Population.new("X", b))
    _P = np.column_stack([np.arange(n), np.arange(n) + n])

    if c is not None:
        _pop = Population.merge(_pop, Population.new("X", c))
        _P = np.column_stack([_P, np.arange(n) + 2 * n])

    problem = get_problem_func(a.shape[1], xl, xu, type_var)(**kwargs)
    return crossover.do(problem, _pop, _P, **kwargs).get("X")


def mutation(mutation, X, xl=0, xu=1, type_var=np.double, **kwargs):
    problem = get_problem_func(X.shape[1], xl, xu, type_var)(**kwargs)
    return mutation.do(problem, Population.new("X", X), **kwargs).get("X")


