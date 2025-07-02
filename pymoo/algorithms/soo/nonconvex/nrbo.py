"""
Newton-Raphson-based optimizer (NRBO)

-------------------------------- Description -------------------------------



-------------------------------- References --------------------------------

[1]. Sowmya, R., Premkumar, M. & Jangir, P. Newton-Raphson-based optimizer:
A new population-based metaheuristic algorithm for continuous optimization problems.
Engineering Applications of Artificial Intelligence 128, 107532 (2024).

----------------------------------------------------------------------------
"""

import numpy as np

from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair
from pymoo.core.replacement import ImprovementReplacement
from pymoo.core.survival import Survival
from pymoo.operators.repair.bounds_repair import repair_random_init
from pymoo.operators.sampling.lhs import LHS
from pymoo.util import default_random_state


class FitnessSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=False)

    def _do(self, problem, pop, n_survive=None, **kwargs):
        F, cv = pop.get("F", "cv")
        assert F.shape[1] == 1, "FitnessSurvival can only used for single objective single!"
        S = np.lexsort([F[:, 0], cv])
        pop.set("rank", np.argsort(S))
        return pop[S[:n_survive]]


@default_random_state
def Search_Rule(Xb, Xw, Xn, rho, random_state=None):
    dim = len(Xn)

    dx = random_state.random(dim) * np.abs(Xb - Xn)

    tmp = Xw + Xb - 2 * Xn
    idx = np.where(tmp == 0.0)
    # repair if xj=0
    if idx:
        tmp[idx] = tmp[idx] + 1e-12
    nrsr = random_state.standard_normal() * (((Xw - Xb) * dx) / (2 * tmp))
    Z = Xn - nrsr

    r1 = random_state.random()
    # r2 = random_state.random()
    tmp = np.mean(Z + Xn)

    yw = r1 * (tmp + r1 * dx)
    yb = r1 * (tmp - r1 * dx)

    NRSR = random_state.standard_normal() * ((yw - yb) * dx) / (2 * (yw + yb - 2 * Xn))

    step = NRSR - rho
    X1 = Xn - step
    X2 = Xb - step
    return X1, X2


class NRBO(Algorithm):
    def __init__(
        self,
        pop_size=50,
        deciding_factor=0.6,
        sampling=LHS(),
        max_iteration=100,
        repair=NoRepair(),
        output=None,
        display=None,
        callback=None,
        archive=None,
        return_least_infeasible=False,
        save_history=False,
        verbose=False,
        seed=None,
        evaluator=None,
        **kwargs,
    ):
        self.max_iteration = max_iteration
        termination = ("n_gen", self.max_iteration)
        self.pop_size = pop_size
        self.deciding_factor = deciding_factor
        self.repair = repair
        self.survial = FitnessSurvival()
        self.initialization = Initialization(sampling, self.repair)
        super().__init__(
            termination,
            output,
            display,
            callback,
            archive,
            return_least_infeasible,
            save_history,
            verbose,
            seed,
            evaluator,
            **kwargs,
        )

    def _setup(self, problem, **kwargs):
        return super()._setup(problem, **kwargs)

    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.pop_size, algorithm=self, random_state=self.random_state)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = self.survial.do(self.problem, infills)

    def _infill(self):
        delta = (1 - (2 * self.n_iter) / self.max_iteration) ** 5

        # find Xb, Xw inviduals
        rank = self.pop.get("rank")
        Xb_idx = np.argmin(rank)
        X = self.pop.get("X")
        Xb = X[Xb_idx]
        Xw_idx = np.argmax(rank)
        Xw = X[Xw_idx]

        off = []

        for i in range(self.pop_size):

            # random select r1,r2
            idx = np.arange(self.pop_size)
            idx = np.delete(idx, i)
            r1, r2 = self.random_state.choice(idx, size=2, replace=False)

            a, b = self.random_state.random(2)
            rho = a * (Xb - X[i]) + b * (X[r1] - X[r2])

            # NRSR
            X1, X2 = Search_Rule(Xb=Xb, Xw=Xw, Xn=X[i], rho=rho, random_state=self.random_state)

            X3 = X[i] - delta * (X2 - X1)

            r2 = self.random_state.random()
            Xn_new = r2 * (r2 * X1 + (1 - r2) * X2) + (1 - r2) * X3

            # TAO
            if self.random_state.random() < self.deciding_factor:
                theta1 = self.random_state.uniform(-1, 1, 1)
                theta2 = self.random_state.uniform(-0.5, 0.5, 1)

                beta = 0 if self.random_state.random() > 0.5 else 1
                u1 = beta * 3 * self.random_state.random() + (1 - beta)
                u2 = beta * self.random_state.random() + (1 - beta)

                tmp = theta1 * (u1 * Xb - u2 * X[i]) + theta2 * delta * (u1 * np.mean(X[i]) - u2 * X[i])
                if u1 < 0.5:
                    X_tao = Xn_new + tmp
                else:
                    X_tao = Xb + tmp

                Xn_new = X_tao
            off.append(Xn_new)

        off = np.array(off)
        if self.problem.has_bounds():
            # off = set_to_bounds_if_outside(off, *self.problem.bounds())
            off = repair_random_init(off, X, *self.problem.bounds(), random_state=self.random_state)

        off = Population.new(X=off)

        off = self.repair.do(self.problem, off)
        return off

    def _advance(self, infills=None, **kwargs):
        off = infills
        has_improved = ImprovementReplacement().do(self.problem, self.pop, off, return_indices=True)

        self.pop[has_improved] = off[has_improved]
        self.survial.do(self.problem, self.pop)

    def _set_optimum(self):
        k = self.pop.get("rank") == 0
        self.opt = self.pop[k]


