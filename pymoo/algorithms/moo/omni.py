import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.core.survival import Survival, split_by_feasibility
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort


def calc_omni_crowding(F, X):
    """
    Combined crowding distance in objective and decision variable space.

    Both spaces are normalized internally (each dimension divided by its range),
    so the resulting distances are comparable regardless of scale or number of
    dimensions. The crowding of each individual is the minimum of its crowding
    in F-space and X-space, ensuring diversity is maintained in both.

    Parameters
    ----------
    F : ndarray, shape (n, n_obj)
        Objective values.
    X : ndarray, shape (n, n_var)
        Decision variable values.

    Returns
    -------
    crowding : ndarray, shape (n,)
        Combined crowding distances (higher = more isolated = more desirable).
    """
    cd_F = calc_crowding_distance(F)

    # Normalize X to [0, 1] before computing crowding so dimension count
    # differences between F and X don't distort the comparison.
    X_range = X.max(axis=0) - X.min(axis=0)
    X_range[X_range == 0] = 1.0
    X_norm = (X - X.min(axis=0)) / X_range
    cd_X = calc_crowding_distance(X_norm)

    return np.minimum(cd_F, cd_X)


class OmniRankAndCrowding(Survival):
    """
    Survival operator for the Omni-Optimizer.

    Differs from standard RankAndCrowding in two ways:
    1. The epsilon used for non-dominated sorting is computed dynamically
       each generation from the current population's objective range:
       epsilon_k = (f_k_max - f_k_min) / (N - 1).
    2. Crowding distance is the minimum of crowding in objective space
       and crowding in decision variable space, promoting diversity in both.

    Reference
    ---------
    Deb, K. & Tiwari, S. (2005). Omni-optimizer: A generic evolutionary
    algorithm for single and multi-objective optimization. GECCO 2005.
    """

    def __init__(self):
        super().__init__(filter_infeasible=True)

    def _do(self, problem, pop, *args, n_survive=None, random_state=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        X = pop.get("X").astype(float, copy=False)
        n = len(pop)

        if n_survive is None:
            n_survive = n

        # Dynamic epsilon: one value per objective based on current spread
        f_range = F.max(axis=0) - F.min(axis=0)
        epsilon = np.where(f_range > 1e-10, f_range / max(n - 1, 1), 0.0)

        fronts = NonDominatedSorting(epsilon=epsilon).do(F, n_stop_if_ranked=n_survive)

        survivors = []

        for k, front in enumerate(fronts):
            I = np.arange(len(front))

            if len(survivors) + len(front) > n_survive:
                n_remove = len(survivors) + len(front) - n_survive

                crowding_of_front = calc_omni_crowding(F[front], X[front])

                I = randomized_argsort(crowding_of_front, order='descending',
                                       method='numpy', random_state=random_state)
                I = I[:-n_remove]
            else:
                crowding_of_front = calc_omni_crowding(F[front], X[front])

            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            survivors.extend(front[I])

        return pop[survivors]


class OmniOptimizer(NSGA2):
    """
    Omni-Optimizer: a generic evolutionary algorithm for single and
    multi-objective optimization that maintains diversity in both objective
    and decision variable space.

    Key differences from NSGA-II:
    - Non-dominated sorting uses a *dynamically computed* epsilon that
      relaxes dominance based on the spread of the current population,
      preventing too many fronts on degenerate landscapes.
    - Crowding distance is computed in both objective and decision variable
      space. Each individual's crowding is the *minimum* of the two,
      so selection pressure preserves spread in both spaces simultaneously.

    This makes the algorithm well-suited for problems with multiple Pareto
    subsets (e.g. multimodal multi-objective problems), where NSGA-II would
    converge to a single subset.

    Parameters
    ----------
    pop_size : int
        Population size. Defaults to 100.
    sampling : Sampling
        Sampling strategy. Defaults to FloatRandomSampling().
    selection : Selection
        Mating selection. Defaults to binary tournament using rank and
        combined crowding.
    crossover : Crossover
        Crossover operator. Defaults to SBX(eta=15, prob=0.9).
    mutation : Mutation
        Mutation operator. Defaults to PM(eta=20).

    References
    ----------
    Deb, K. & Tiwari, S. (2005). Omni-optimizer: A generic evolutionary
    algorithm for single and multi-objective optimization. GECCO 2005.
    """

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 output=MultiObjectiveOutput(),
                 **kwargs):

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=OmniRankAndCrowding(),
                         output=output,
                         **kwargs)
