"""
This class provide an interface for other libraries to specific modules. For example, the evolutionary operations
can be used easily just by calling a function and providing the lower and upper bounds of the problem.

"""
import copy
import types

import numpy as np

from pymoo.model.algorithm import filter_optimum
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.model.problem import Problem


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


# =========================================================================================================
# Ask And Tell Interface
# =========================================================================================================


def evaluate_to_nan(self, x, out, *args, **kwargs):
    n_points, _ = x.shape
    out["F"] = None
    if self.n_constr > 0:
        out["G"] = None


def evaluate_to_value(F, G=None):
    def eval(self, x, out, *args, **kwargs):
        n_points, _ = x.shape
        out["F"] = F
        if G is not None:
            out["G"] = G

    return eval


class AskAndTell:

    def __init__(self, algorithm, problem=None, **kwargs):

        if problem is not None:
            self.problem = copy.deepcopy(problem)
        else:
            self.problem = Problem(**kwargs)

        self.algorithm = copy.deepcopy(algorithm)

    def get_population(self):
        return self.algorithm.pop

    def set_population(self, pop):
        self.algorithm.pop = pop

    def get_offsprings(self):
        return self.algorithm.off

    def set_offsprings(self, off):
        self.algorithm.off = off

    def ask(self):

        # if the initial population has not been generated yet
        if self.get_population() is None:

            self.algorithm.setup(self.problem)

            # deactivate the survival because no values have been set yet
            survival = self.algorithm.survival
            self.algorithm.survival = None

            self.problem._evaluate = types.MethodType(evaluate_to_nan, self.problem)
            self.algorithm._initialize()

            # activate the survival for the further runs
            self.algorithm.survival = survival

            return self.get_population().get("X")

        # usually the case - create the next output
        else:

            # if offsprings do not exist set the pop - otherwise always offsprings
            if self.get_offsprings() is not None:
                self.set_population(Population.merge(self.get_population(), self.get_offsprings()))

            # execute a survival of the algorithm
            survivors = self.algorithm.survival.do(self.problem, self.get_population(),
                                                   self.algorithm.pop_size, algorithm=self.algorithm)
            self.set_population(survivors)

            # execute the mating using the population
            off = self.algorithm.mating.do(self.algorithm.problem, self.get_population(),
                                           n_offsprings=self.algorithm.n_offsprings, algorithm=self.algorithm)

            # execute the fake evaluation of the individuals
            self.problem._evaluate = types.MethodType(evaluate_to_nan, self.problem)
            self.algorithm.evaluator.eval(self.problem, off, algorithm=self.algorithm)
            self.set_offsprings(off)

            return off.get("X")

    def tell(self, F, G=None, X=None):

        # if offsprings do not exist set the pop - otherwise always offsprings
        pop_to_evaluate = self.get_offsprings() if self.get_offsprings() is not None else self.get_population()

        # if the user changed the design space values for whatever reason
        if X is not None:
            pop_to_evaluate.set("X")

        # do the function evaluations
        self.problem._evaluate = types.MethodType(evaluate_to_value(F.copy(), G.copy()), self.problem)
        self.algorithm.evaluator.eval(self.problem, pop_to_evaluate, algorithm=self.algorithm)

    def result(self, only_optimum=True, return_values_of="auto"):

        if return_values_of == "auto":
            return_values_of = ["X", "F"]
            if self.problem.n_constr > 0:
                return_values_of.append("CV")

        if only_optimum:
            self.algorithm.finalize()
            pop, opt = self.algorithm.pop, self.algorithm.opt
            res = filter_optimum(pop.copy()) if opt is None else opt.copy()

            if isinstance(res, Individual):
                res = Population.create(res)

        else:
            res = self.algorithm.pop

        return res.get(*return_values_of)


