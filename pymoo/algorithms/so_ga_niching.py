import numpy as np

from pymoo.algorithms.so_genetic_algorithm import GA, FitnessSurvival
from pymoo.docs import parse_doc_string
from pymoo.model.survival import Survival
from pymoo.operators.selection.tournament_selection import compare, TournamentSelection
from pymoo.util.clearing import EpsilonClearing
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.misc import norm_eucl_dist
from pymoo.util.termination.constr_violation import ConstraintViolationToleranceTermination
from pymoo.util.termination.default import SingleObjectiveDefaultTermination, DefaultTermination
from pymoo.util.termination.f_tol_single import SingleObjectiveSpaceToleranceTermination
from pymoo.util.termination.x_tol import DesignSpaceToleranceTermination


# =========================================================================================================
# Display
# =========================================================================================================

class NicheDisplay(SingleObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.attrs = [e for e in self.output.attrs if e[0] != "favg"]
        self.output.append("n_niches", len(algorithm.opt), width=10)
        self.output.append("favg", algorithm.opt.get("F").mean(), width=12)


# =========================================================================================================
# Termination
# =========================================================================================================

class NicheSingleObjectiveSpaceToleranceTermination(SingleObjectiveSpaceToleranceTermination):

    def _store(self, algorithm):
        return algorithm.opt.get("F").mean()


class NicheTermination(DefaultTermination):

    def __init__(self,
                 x_tol=1e-32,
                 cv_tol=1e-6,
                 f_tol=1e-6,
                 nth_gen=5,
                 n_last=20,
                 **kwargs) -> None:
        super().__init__(DesignSpaceToleranceTermination(tol=x_tol, n_last=n_last),
                         ConstraintViolationToleranceTermination(tol=cv_tol, n_last=n_last),
                         NicheSingleObjectiveSpaceToleranceTermination(tol=f_tol, n_last=n_last, nth_gen=nth_gen),
                         **kwargs)


# =========================================================================================================
# Selection
# =========================================================================================================


def comp_by_cv_and_clearing_fitness(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV,
                           method='smaller_is_better',
                           return_random_if_equal=True)

        # first compare by the round the individual was selected
        else:
            S[i] = compare(a, pop[a].get("iter"), b, pop[b].get("iter"), method='smaller_is_better')

            # if it was the same round - then use the rank of the fitness directly
            if np.isnan(S[i]):
                S[i] = compare(a, pop[a].get("rank"), b, pop[b].get("rank"),
                               method='smaller_is_better', return_random_if_equal=True)

    return S[:, None].astype(int)


# =========================================================================================================
# Survival
# =========================================================================================================

class EpsilonClearingSurvival(Survival):

    def __init__(self, epsilon, n_max_each_iter=None) -> None:
        super().__init__(False)
        self.epsilon = epsilon
        self.n_max_each_iter = n_max_each_iter

    def _do(self, problem, pop, n_survive, out=None, **kwargs):
        F = pop.get("F")

        if F.shape[1] != 1:
            raise ValueError("FitnessSurvival can only used for single objective single!")

        # this basically sorts the population by constraint and objective value
        pop = FitnessSurvival().do(problem, pop, len(pop))

        # calculate the distance from each individual to another - pre-processing for the clearing
        X = pop.get("X")
        D = norm_eucl_dist(problem, X, X)

        # initialize the clearing strategy
        clearing = EpsilonClearing(D, self.epsilon)

        # initialize the iteration and rank i the beginning
        iter, rank = 1, 1

        # also solutions that have been found in the first iteration
        iter_one = None

        # until the number of selected individuals are less than expected survivors
        while len(clearing.selected()) < n_survive:

            # get all the remaining indices
            remaining = clearing.remaining()

            # if no individuals are left because of clearing - perform a reset
            if len(remaining) == 0 or (self.n_max_each_iter is not None and rank > self.n_max_each_iter):
                # reset and retrieve the newly available indices
                clearing.reset()
                remaining = clearing.remaining()

                # increase the iteration counter and start over from rank 1
                iter += 1
                rank = 1

                # get the individual of the first iteration - needed for niche assignment
                iter_one = np.where(pop.get("iter") == 1)[0] if iter_one is None else iter_one

            # since the population is ordered by F and CV it is always the first index
            k = remaining[0]

            # set the attribute to the selected individual
            pop[k].set("iter", iter)
            pop[k].set("rank", rank)

            # in the first iteration set the niche counter for each solution equal to rank
            if iter == 1:
                pop[k].set("niche", rank)
            else:
                closest_iter_one = iter_one[D[k][iter_one].argmin()]
                niche = pop[closest_iter_one].get("niche")
                pop[k].set("niche", niche)

            clearing.select(k)
            rank += 1

        # retrieve all individuals being selected
        S = clearing.selected()

        return pop[S]


# =========================================================================================================
# Algorithm
# =========================================================================================================


class NicheGA(GA):

    def __init__(self,
                 pop_size=100,
                 norm_niche_size=0.05,
                 display=NicheDisplay(),
                 **kwargs):
        """

        Parameters
        ----------
        norm_niche_size : float
            The radius in which the clearing shall be performed. The clearing is performed in the normalized design
            space, e.g. 0.05 corresponds to clear all solutions which have less norm euclidean distance than 5%.
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """

        surv = kwargs.get("survival")
        if surv is None:
            surv = EpsilonClearingSurvival(norm_niche_size, n_max_each_iter=None)

        selection = kwargs.get("selection")
        if selection is None:
            selection = TournamentSelection(comp_by_cv_and_clearing_fitness)

        super().__init__(pop_size=pop_size,
                         selection=selection,
                         survival=surv,
                         display=display,
                         **kwargs)

        # self.default_termination = NicheTermination()
        self.default_termination = SingleObjectiveDefaultTermination()

    def _set_optimum(self, **kwargs):
        self.opt = self.pop[self.pop.get("iter") == 1]


parse_doc_string(NicheGA.__init__)
