"""
This class provide an interface for other libraries to specific modules. For example, the evolutionary operations
can be used easily just by calling a function and providing the lower and upper bounds of the problem.

"""
import copy
import types
import numpy as np

from pymoo.factory import get_problem
from pymoo.model.evaluator import Evaluator
from pymoo.model.population import Population
from pymoo.model.problem import Problem


def get_problem_func(n_var, xl, xu, type_var):
    class P(Problem):
        def __init__(self) -> None:
            super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=xl, xu=xu, type_var=type_var)

    return P


def sample(sampling, n_samples, n_var, xl=0, xu=1, type_var=np.double, **kwargs):
    problem = get_problem_func(n_var, xl, xu, type_var)(**kwargs)
    return sampling.do(problem, Population(), n_samples, **kwargs).get("X")


def crossover(crossover, a, b, c=None, xl=0, xu=1, type_var=np.double, **kwargs):
    n = a.shape[0]
    _pop = Population().new("X", a).merge(Population().new("X", b))
    _P = np.column_stack([np.arange(n), np.arange(n) + n])

    if c is not None:
        _pop = _pop.merge(Population().new("X", c))
        _P = np.column_stack([_P, np.arange(n) + 2 * n])

    problem = get_problem_func(a.shape[1], xl, xu, type_var)(**kwargs)
    return crossover.do(problem, _pop, _P, **kwargs).get("X")


def mutation(mutation, X, xl=0, xu=1, type_var=np.double, **kwargs):
    problem = get_problem_func(X.shape[1], xl, xu, type_var)(**kwargs)
    return mutation.do(problem, Population().new("X", X), **kwargs).get("X")


# below implements the ask&tell api
class AskAndTell:
    """ ask&tell api wrapper of pymoo"""
    def __init__(self,
                 problem,
                 method,
                 popsize=100,  # population size
                 ):
        self.method = method
        # hacky approach
        def aux_func(self, x, out, *args, **kwargs):
            out["F"] = np.full((x.shape[0], self.n_obj), np.nan)
            if self.n_constr > 0:
                out["G"] = np.full((x.shape[0], self.n_constr), np.nan)

        self.method.evaluator = Evaluator()
        self.method.problem = copy.deepcopy(problem)
        self.method.problem._evaluate = types.MethodType(aux_func, self.method.problem)
        self.popsize = popsize
        self.pop = None
        self.off = None

    def ask(self):
        """ return a group of solution candidates of size popsize"""
        if self.pop is None:
            self.pop = self.method._initialize()
            return self.pop.get("X")
        else:
            # survival -> mating
            if self.off is not None:
                pop = self.pop.merge(self.off)
            else:
                pop = self.pop
            self.pop = self.method.survival.do(self.method.problem, pop, self.popsize, algorithm=self.method)
            self.off = self.method._mating(self.pop)
            # has to call to set a few required attributes
            self.method.evaluator.eval(self.method.problem, self.off, algorithm=self.method)
            return self.off.get('X')

    def tell(self, obj, constr=None):
        """ setting the user provided f back to """
        assert (len(obj) == self.popsize), "Inconsistent objective size reported."
        # always make a copy of f and g to prevent bug
        f = np.copy(obj)

        if self.off is None:
            self.pop.set("F", f)
            if constr is not None:
                g = np.copy(constr)
                cv = Problem.calc_constraint_violation(g)
                self.pop.set("G", g)
                self.pop.set("CV", cv)
                self.pop.set("feasible", (cv <= 0))
        else:
            self.off.set("F", f)
            if constr is not None:
                g = np.copy(constr)
                cv = Problem.calc_constraint_violation(g)
                self.off.set("G", g)
                self.off.set("CV", cv)
                self.off.set("feasible", (cv <= 0))

    def result(self):
        """ report the current best solutions """
        if self.method.problem.n_obj > 1:
            non_dominated = self.pop.get("rank") == 0
            return (self.pop.get("X")[non_dominated, :],
                    self.pop.get("F")[non_dominated, :])
        else:
            if self.method.problem.n_constr > 0:
                feasible = self.pop.get("feasible")
            else:
                feasible = np.ones(self.popsize, dtype=np.bool)
            elite = np.argmin(self.pop.get("F")[feasible, 0])
            return (self.pop.get("X")[feasible, :][elite, :],
                    self.pop.get("F")[feasible, 0][elite])

if __name__ == "__main__":
    # ----- debug for ask&tell API ----------
    from pymoo.algorithms.so_genetic_algorithm import ga

    method = ga(pop_size=200, eliminate_duplicates=True)
    problem = get_problem('g01')
    api = AskAndTell(problem=problem, method=method, popsize=200)

    f = np.full((api.popsize, problem.n_obj), np.full)
    g = np.full((api.popsize, problem.n_constr), np.full)

    for gen in range(1, 201):
        candidates = api.ask()
        res = {}
        problem._evaluate(candidates, out=res)
        f[:, 0] = res["F"]
        g[:, :] = res["G"]
        api.tell(f, g)
        print(np.min(api.pop.get("F")))



    # ----- end of debug --------------------

    #
    # from pymoo.interface import crossover
    # from pymoo.factory import get_crossover
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    #
    # def example_parents(n_matings, n_var):
    #     a = np.arange(n_var)[None, :].repeat(n_matings, axis=0)
    #     b = a + n_var
    #     return a, b
    #
    #
    # def show(M):
    #     plt.figure(figsize=(4, 4))
    #     plt.imshow(M, cmap='Greys', interpolation='nearest')
    #     plt.show()
    #
    #
    # n_matings, n_var = 100, 100
    # a, b = example_parents(n_matings, n_var)
    #
    # print("One Point Crossover")
    # off = crossover(get_crossover("bin_one_point"), a, b)
    # show((off[:n_matings] != a[0]))
    #
    # print("Two Point Crossover")
    # off = crossover(get_crossover("bin_two_point"), a, b)
    # show((off[:n_matings] != a[0]))
    # show((off[n_matings:] != b[0]))
    # print(off[n_matings:])
    #
    # print("K Point Crossover (k=4)")
    # off = crossover(get_crossover("bin_k_point", n_points=4), a, b)
    # show((off[:n_matings] != a[0]))
    #
