import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection, compare
from pymoo.util.display import disp_single_objective
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize


# =========================================================================================================
# Implementation
# =========================================================================================================


class SingleObjectiveGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, **kwargs):
        set_if_none(kwargs, 'pop_size', 100)
        set_if_none(kwargs, 'sampling', RandomSampling())
        set_if_none(kwargs, 'selection', TournamentSelection(func_comp=comp_by_cv_and_fitness))
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=0.9, eta=3))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=None, eta=5))
        set_if_none(kwargs, 'survival', FitnessSurvival())
        set_if_none(kwargs, 'eliminate_duplicates', True)

        super().__init__(**kwargs)
        self.func_display_attrs = disp_single_objective


class FitnessSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(True)

    def _do(self, pop, n_survive, out=None, **kwargs):
        F = pop.get("F")

        if F.shape[1] != 1:
            raise ValueError("FitnessSurvival can only used for single objective problems!")

        return pop[np.argsort(F[:, 0])[:n_survive]]


class ConstraintHandlingSurvival(Survival):

    def __init__(self, method="parameter_less", **kwargs) -> None:
        super().__init__(False)
        self.method = method
        self.params = kwargs

        self.min_constraints = None
        self.max_constraints = None

    def calc_normalized_constraints(self, G):

        # update the ideal point for constraints
        if self.min_constraints is None:
            self.min_constraints = np.full(G.shape[1], np.inf)
        self.min_constraints = np.min(np.vstack((self.min_constraints, G)), axis=0)

        # update the nadir point for constraints
        non_dominated = NonDominatedSorting().do(G, return_rank=True, only_non_dominated_front=True)

        if self.max_constraints is None:
            self.max_constraints = np.full(G.shape[1], np.inf)
        self.max_constraints = np.min(np.vstack((self.max_constraints, np.max(G[non_dominated, :], axis=0))), axis=0)

        return normalize(G, self.min_constraints, self.max_constraints)

    def _do(self, pop, n_survive, out=None, **kwargs):

        # check if it is a population with a single objective
        F, G = pop.get("F", "G")
        if F.shape[1] != 1:
            raise ValueError("FitnessSurvival can only used for single objective problems!")

        # default parameters if not provided to the algorithm
        DEFAULT_PARAMS = {
            "parameter_less": {},
            "epsilon_constrained": {"epsilon": 1e-2},
            "penalty": {"weight": 0.1},
            "stochastic_ranking": {"weight": 0.45},
        }

        # check if the method is known
        if self.method not in DEFAULT_PARAMS.keys():
            raise Exception("Unknown constraint handling method %s" % self.method)

        # set the default parameter if not provided
        for key, value in DEFAULT_PARAMS[self.method].items():
            set_if_none(self.params, key, value)

        # make the lowest possible constraint violation 0 - if not violated in that constraint
        G = G * (G > 0).astype(np.float)

        # find value to normalize to sum of for CV
        for j in range(G.shape[1]):

            N = np.median(G[:, j])
            if N == 0:
                N = np.max(G[:, j])

            if N > 0:
                pass
                # G[:, j] /= N

        # add the constraint violation and divide by normalization factor
        CV = np.sum(G, axis=1)

        if self.method == "parameter_less":

            # if infeasible add the constraint violation to worst F value
            _F = np.max(F, axis=0) + CV
            infeasible = CV > 0

            F[infeasible, 0] = _F[infeasible]

            # do fitness survival as done before with modified f
            return pop[np.argsort(F[:, 0])[:n_survive]]

        elif self.method == "epsilon_constrained":

            _F = np.max(F, axis=0) + CV
            infeasible = CV > self.params["epsilon"]
            F[infeasible, 0] = _F[infeasible]

            # do fitness survival as done before with modified f
            return pop[np.argsort(F[:, 0])[:n_survive]]

        elif self.method == "penalty":

            _F = normalize(F)

            # add for each constraint violation a penalty
            _F[:, 0] = _F[:, 0] + self.params["weight"] * CV
            return pop[np.argsort(_F[:, 0])[:n_survive]]

        elif self.method == "stochastic_ranking":

            # first shuffle the population randomly - to be sorted again
            I = np.random.permutation(len(pop))
            pop, F, CV = pop[I], F[I], CV[I]

            # func = load_function("stochastic_ranking", "stochastic_ranking")

            from stochastic_ranking import stochastic_ranking
            func = stochastic_ranking

            index = func(F[:, 0], CV, self.params["prob"])

            return pop[index[:n_survive]]


def comp_by_cv_and_fitness(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible just set random
        else:
            S[i] = compare(a, pop[a].F, b, pop[b].F, method='smaller_is_better', return_random_if_equal=True)

    return S[:, None].astype(np.int)


# =========================================================================================================
# Interface
# =========================================================================================================


def ga(
        pop_size=100,
        sampling=RandomSampling(),
        selection=TournamentSelection(func_comp=comp_by_cv_and_fitness),
        crossover=SimulatedBinaryCrossover(prob=0.9, eta=3),
        mutation=PolynomialMutation(prob=None, eta=5),
        eliminate_duplicates=True,
        n_offsprings=None,
        **kwargs):
    """

    Parameters
    ----------
    pop_size : {pop_size}
    sampling : {sampling}
    selection : {selection}
    crossover : {crossover}
    mutation : {mutation}
    eliminate_duplicates : {eliminate_duplicates}
    n_offsprings : {n_offsprings}

    Returns
    -------
    ga : :class:`~pymoo.model.algorithm.Algorithm`
        Returns an SingleObjectiveGeneticAlgorithm algorithm object.


    """

    return SingleObjectiveGeneticAlgorithm(pop_size=pop_size,
                                           sampling=sampling,
                                           selection=selection,
                                           crossover=crossover,
                                           mutation=mutation,
                                           survival=FitnessSurvival(),
                                           eliminate_duplicates=eliminate_duplicates,
                                           n_offsprings=n_offsprings,
                                           **kwargs)


parse_doc_string(ga)
