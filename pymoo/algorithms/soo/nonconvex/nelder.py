from typing import Generator

import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.local import LocalSearch
from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluation
from pymoo.core.fitness import sort_by_fitness, is_better
from pymoo.core.output import SingleObjectiveOutput, Column
from pymoo.core.problem import Problem
from pymoo.core.solution import SolutionSet, merge
from pymoo.core.termination import Termination
from pymoo.util import y_and_n


class NelderAndMeadOutput(SingleObjectiveOutput):

    def setup(self, _):
        super().setup(_)
        self.columns += [
            Column(name='status', f_get=lambda a: a.status, width=12),
            Column(name='x_delta', f_get=lambda a: a.x_delta, width=9),
            Column(name='f_delta', f_get=lambda a: a.f_delta, width=9),
            Column(name='is_degenerated', f_get=lambda a: y_and_n(a.is_degenerated), width=9)
        ]


class NelderAndMeadTermination(Termination):

    def __init__(self,
                 x_tol: float = 1e-7,
                 f_tol: float = 1e-7):
        super().__init__()
        self.x_tol = x_tol
        self.f_tol = f_tol

    def update(self, algorithm):
        sols, problem = algorithm.sols, algorithm.problem

        if len(sols) <= 1:
            return 0.0
        f_tol = 1 / (1 + (algorithm.f_delta - self.f_tol))
        x_tol = 1 / (1 + (algorithm.x_delta - self.x_tol))
        is_degenerated = int(algorithm.is_degenerated)
        self.status = max(f_tol, x_tol, is_degenerated)


class NelderMead(LocalSearch):

    def __init__(self,
                 termination=NelderAndMeadTermination(),
                 output=NelderAndMeadOutput(),
                 **kwargs):
        super().__init__(termination=termination, output=output, **kwargs)
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.delta = None

        self.init_simplex = None
        self.status = None

        self.x_delta = None
        self.f_delta = None
        self.is_degenerated = None

    def setup(self, problem: Problem, **kwargs) -> Algorithm:
        super().setup(problem, **kwargs)
        self.problem = problem

        vtype = problem.vtype

        n = vtype.size
        self.alpha = 1
        self.beta = 1 + 2 / n
        self.gamma = 0.75 - 1 / (2 * n)
        self.delta = 1 - 1 / n

        return self

    def initialize(self) -> Generator[Evaluation, None, SolutionSet]:
        yield from super().initialize()

        init_sol = self.init_sol
        init_simplex = create_simplex(init_sol.x, bounds=self.problem.vtype.bounds)
        sols = yield from self.evaluator.send(init_simplex)

        return merge(init_sol, sols)

    def advance(self) -> Generator[Evaluation, None, SolutionSet]:
        vtype = self.problem.vtype
        n = vtype.size - 1
        pop = self.sols

        # calculate the centroid
        centroid = pop[:n + 1].get("X").mean(axis=0)

        # REFLECT #

        # reflect the point, consider factor if bounds are there, make sure in bounds (floating point) evaluate
        reflect = yield from self.evaluator.send(centroid + self.alpha * (centroid - pop[n + 1].X))
        self.status = 'REFLECT'

        # whether a shrink is necessary or not - decided during this step
        shrink = False

        better_than_current_best = is_better(reflect, pop[0])
        better_than_second_worst = is_better(reflect, pop[n])
        better_than_worst = is_better(reflect, pop[n + 1])

        # if better than the current best - check for expansion
        if better_than_current_best:

            # EXPAND
            expand = yield from self.evaluator.send(centroid + self.beta * (reflect.x - centroid))

            # if the expansion further improved take it - otherwise use expansion
            if is_better(expand, reflect):
                pop[n + 1] = expand
                self.status = 'EXPAND'
            else:
                pop[n + 1] = reflect

        # if the new point is not better than the best, but better than second worst - just keep it
        elif not better_than_current_best and better_than_second_worst:
            pop[n + 1] = reflect

        # if not worse than the worst - outside contraction
        elif not better_than_second_worst and better_than_worst:

            # Outside Contraction
            contract_outside = yield from self.evaluator.send(centroid + self.gamma * (reflect.x - centroid))

            if is_better(contract_outside, reflect):
                pop[n + 1] = contract_outside
                self.status = 'CONTRACT_OUT'
            else:
                shrink = True

        # if the reflection was worse than the worst - inside contraction
        else:
            # Inside Contraction #
            contract_inside = yield from self.evaluator.send(centroid - self.gamma * (reflect.x - centroid))

            if is_better(contract_inside, pop[n + 1]):
                pop[n + 1] = contract_inside
                self.status = 'CONTRACT_IN'
            else:
                shrink = True

        # Shrink #
        if shrink:
            x_best, x_others = pop[0].X, pop[1:].get("X")
            x_shrink = x_best + self.delta * (x_others - x_best)
            pop[1:] = yield from self.evaluator.send(x_shrink)
            self.status = 'SHRINK'

        # sort the population by te objective value
        pop = sort_by_fitness(pop)

        # store information about the current simplex
        X, F = pop.get("X", "F")
        norm = self.problem.vtype.norm()
        self.x_delta = np.abs((X[1:] - X[0]) / norm).max()
        self.f_delta = np.abs(F[1:] - F[0]).max()

        # degenerated simplex - get all edges and minimum and maximum length
        D = cdist(X, X)
        val = D[np.triu_indices(len(X), 1)]
        min_e, max_e = val.min(), val.max()

        # either if the maximum length is very small or the ratio is degenerated
        self.is_degenerated = max_e < 1e-16 or min_e / max_e < 1e-16

        return pop


def create_simplex(init_x, bounds=None, scale=0.05):
    n = len(init_x)

    if bounds is not None:
        low, high = bounds
        delta = scale * (high - low)
    else:
        delta = scale * init_x
        delta[delta == 0] = 0.00025

    # repeat the x0 already and add the values
    X = init_x[None, :].repeat(n, axis=0)

    for k in range(n):

        # if the problem has bounds do the check
        if bounds is not None:
            low, high = bounds
            if X[k, k] + delta[k] < high[k]:
                X[k, k] = X[k, k] + delta[k]
            else:
                X[k, k] = X[k, k] - delta[k]

        # otherwise just add the init_simplex_scale
        else:
            X[k, k] = X[k, k] + delta[k]

    return X
