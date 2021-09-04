import numpy as np

from pymoo.algorithms.soo.nonconvex.de import DifferentialEvolutionMating
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_problem
from pymoo.core.population import Population
from pymoo.core.replacement import hierarchical_sort
from pymoo.operators.crossover.pcx import PCX
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.optimize import minimize


# =========================================================================================================
# Implementation
# =========================================================================================================


def constr_binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f, = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        a_below, b_below = pop[a].get("below"), pop[b].get("below")

        if a_cv > 0.0 and b_cv > 0.0:
            if a_below and b_below:
                S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)
            elif not a_below and not b_below:
                S[i] = compare(a, a_f, b, b_f, method='smaller_is_better', return_random_if_equal=True)
            elif a_below and not b_below:
                S[i] = a
            elif not a_below and b_below:
                S[i] = b

        elif a_cv == 0.0 and b_cv == 0.0:
            S[i] = np.random.choice([a, b])
            # S[i] = compare(a, a_f, b, b_f, method='smaller_is_better', return_random_if_equal=True)
        else:
            S[i] = np.random.choice([a, b])

    return S[:, None].astype(int, copy=False)


class ConstrGA(GA):

    def __init__(self,
                 selection=TournamentSelection(func_comp=constr_binary_tournament),
                 **kwargs):
        super().__init__(selection=selection, advance_after_initial_infill=True, **kwargs)

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."

        pop = Population.merge(self.pop, infills)
        F, CV = pop.get("F", "CV")
        f, cv = F[:, 0, ], CV[:, 0]

        S = hierarchical_sort(f, cv)
        pop, f, cv, feas = pop[S], f[S], cv[S], cv[S] <= 0

        survivors = list(range(30))
        n_remaining = self.pop_size - len(survivors)

        if feas.sum() > 0:
            f_min, cv_min = f[feas].min(), 0.0
        else:
            f_min, cv_min = np.inf, cv[~feas].min()

        I = np.where(np.logical_and(~feas, f < f_min))[0]
        # survivors.extend(I[:n_remaining])
        survivors.extend(I[cv[I].argsort()][:n_remaining])

        if len(survivors) < n_remaining:
            I = np.where(np.logical_and(~feas, f >= f_min))[0]
            survivors.extend(I[f[I].argsort()][:n_remaining])
            # survivors.extend(np.random.choice(I, size=n_remaining))

        if self.n_gen > 1000:
            import matplotlib.pyplot as plt

            plt.scatter(cv[~feas], f[~feas], facecolor="none", edgecolors="red", alpha=0.5, s=20)
            plt.scatter(cv[feas], f[feas], facecolor="none", edgecolors="blue", alpha=0.5, s=40)

            plt.scatter(cv[survivors], f[survivors], color="black", s=3, alpha=0.9)

            plt.show()

        self.pop = pop[survivors]
        self.pop.set("below", self.pop.get("F")[:, 0] <= f_min)

        # print("n_feas", np.sum(pop[survivors].get("CV")[:, 0] <= 0))




problem = get_problem("g09")

algorithm = ConstrGA(pop_size=200,
                     # mating=DifferentialEvolutionMating(variant="DE/best/1/bin")
                     )

res = minimize(problem,
               algorithm,
               ("n_gen", 3000),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
