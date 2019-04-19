"""
This class provide an interface for other libraries to specific modules. For example, the evolutionary operations
can be used easily just by calling a function and providing the lower and upper bounds of the problem.

"""
from pymoo.model.population import Population
from pymop import Problem
import numpy as np


def get_problem_func(n_var, xl, xu, type_var):
    class P(Problem):
        def __init__(self) -> None:
            super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=xl, xu=xu, type_var=type_var)

    return P


def sample(sampling, n_samples, n_var, xl=0, xu=1, type_var=np.double, **kwargs):
    problem = get_problem_func(n_var, xl, xu, type_var)(**kwargs)
    return sampling.sample(problem, Population(), n_samples, **kwargs).get("X")


def crossover(crossover, X, xl=0, xu=1, type_var=np.double, **kwargs):
    problem = get_problem_func(X.shape[1], xl, xu, type_var)(**kwargs)
    return crossover.do(problem, Population(), **kwargs).get("X")


def mutation(mutation, X, xl=0, xu=1, type_var=np.double, **kwargs):
    problem = get_problem_func(X.shape[1], xl, xu, type_var)(**kwargs)
    return mutation.do(problem, Population(), **kwargs).get("X")
