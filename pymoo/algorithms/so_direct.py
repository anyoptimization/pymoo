import numpy as np

from pymoo.algorithms.so_local_search import LocalSearch
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.problems.single import Himmelblau, Sphere, Rastrigin
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize, denormalize


def norm_bounds(pop, problem):
    nxl = normalize(pop.get("xl"), problem.xl, problem.xu)
    nxu = normalize(pop.get("xu"), problem.xl, problem.xu)
    return nxl, nxu


def update_bounds(ind, xl, xu, k, delta):
    _xl = np.copy(xl)
    _xl[k] = ind.X[k] - delta
    ind.set("xl", _xl)

    _xu = np.copy(xu)
    _xu[k] = ind.X[k] + delta
    ind.set("xu", _xu)


class DIRECT(LocalSearch):

    def __init__(self,
                 eps=1e-2,
                 penalty=0.1,
                 display=SingleObjectiveDisplay(),
                 **kwargs):
        super().__init__(display=display, **kwargs)
        self.eps = eps
        self.penalty = penalty

    def initialize(self, problem, **kwargs):
        super().initialize(problem, **kwargs)

        xl, xu = problem.bounds()
        X = denormalize(0.5 * np.ones(problem.n_var), xl, xu)
        x0 = Individual(X=X)
        x0.set("xl", xl)
        x0.set("xu", xu)
        x0.set("depth", 0)
        self.x0 = x0

    def _initialize(self, **kwargs):
        super()._initialize(**kwargs)

    def _potential_optimal(self):
        pop = self.pop

        if len(pop) == 1:
            return pop

        # get the intervals of each individual
        _F, _CV, xl, xu = pop.get("F", "CV", "xl", "xu")
        nF = normalize(_F)
        F = nF + self.penalty * _CV

        # get the length of the interval of each solution
        nxl, nxu = norm_bounds(pop, problem)
        length = (nxu - nxl) / 2

        val = length.max(axis=1)

        # (a) non-dominated with respect to interval
        obj = np.column_stack([-val, F])
        I = NonDominatedSorting().do(obj, only_non_dominated_front=True)
        candidates, F, xl, xu, val = pop[I], F[I], xl[I], xu[I], val[I]

        # import matplotlib.pyplot as plt
        # plt.scatter(obj[:, 0], obj[:, 1])
        # plt.scatter(obj[I, 0], obj[I, 1], color="red")
        # plt.show()

        if len(candidates) == 1:
            return candidates

        else:
            # TODO: The second condition needs to be implemented here. Exact implementation still unclear.

            n_max_candidates = 10

            if len(candidates) > n_max_candidates:
                I = list(np.random.choice(np.arange(len(candidates)), n_max_candidates - 1))
                k = np.argmin(F[:, 0])
                if k not in I:
                    I.append(k)
                candidates = candidates[I]

            return candidates

    def _next(self):
        # the offspring population to finally evaluate and attach to the population
        off = Population()

        # find the potential optimal solution in the current population
        potential_optimal = self._potential_optimal()

        # for each of those solutions execute the division move
        for current in potential_optimal:

            # find the largest dimension the solution has not been evaluated yet
            nxl, nxu = norm_bounds(current, problem)
            k = np.argmax(nxu - nxl)

            # the delta value to be used to get left and right - this is one sixth of the range
            xl, xu = current.get("xl"), current.get("xu")

            delta = (xu[k] - xl[k]) / 6

            # print(current.X, delta, k, xl, xu)

            # create the left individual
            left_x = np.copy(current.X)
            left_x[k] = xl[k] + delta
            left = Individual(X=left_x)

            # create the right individual
            right_x = np.copy(current.X)
            right_x[k] = xu[k] - delta
            right = Individual(X=right_x)

            # update the boundaries for all the points accordingly
            for ind in [current, left, right]:
                update_bounds(ind, xl, xu, k, delta)

            # create the offspring population, evaluate and attach to current population
            _off = Population.create(left, right)
            _off.set("depth", current.get("depth") + 1)

            off = Population.merge(off, _off)

        # evaluate the offsprings
        self.evaluator.eval(self.problem, off, algorithm=self)

        # print(off.get("X"))

        # add the offsprings to the population
        self.pop = Population.merge(self.pop, off)


class ExSwarm(Problem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=1, n_constr=0, xl=np.array([-1.0, -1.0]), xu=np.array([1.0, 1.0]))

    def _evaluate(self, x, out, *args, **kwargs):
        v1 = 20 * x[:, 0]
        v2 = 20 * x[:, 1]
        out["F"] = -np.sin(v1 / np.pi) ** 2. * np.sin(v2 / np.pi) ** 2. * (abs(v1) + abs(v2) + 0.1 * (v1 + v2)) + (
                v1 * v1 + v2 * v2) / 30


if __name__ == '__main__':
    problem = ExSwarm()
    problem = Himmelblau()
    # problem = Sphere(n_var=100, opt=0.2 * np.ones(100))
    problem = Rastrigin(n_var=10)
    problem.xl *= 1.5

    # problem = get_problem("g02")
    algorithm = DIRECT()
    # algorithm = GA()

    ret = minimize(problem,
                   algorithm,
                   ("n_iter", 1000),
                   verbose=True)
