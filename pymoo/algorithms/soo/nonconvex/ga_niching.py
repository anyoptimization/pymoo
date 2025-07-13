import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival, GA
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.cv import ConstraintViolationTermination
from pymoo.termination.default import DefaultSingleObjectiveTermination, DefaultTermination
from pymoo.termination.ftol import SingleObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.termination.xtol import DesignSpaceTermination
from pymoo.util.clearing import EpsilonClearing
from pymoo.util.display.column import Column
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util.misc import norm_eucl_dist


# =========================================================================================================
# Display
# =========================================================================================================

class NicheOutput(SingleObjectiveOutput):

    def __init__(self):
        super().__init__()
        self.n_niches = Column("n_niches", width=10, func=lambda algorithm: len(algorithm.opt))
        self.columns += [self.n_niches]


# =========================================================================================================
# Termination
# =========================================================================================================

class NicheSingleObjectiveSpaceToleranceTermination(SingleObjectiveSpaceTermination):

    def _data(self, algorithm):
        return algorithm.opt.get("F").mean()


class NicheTermination(DefaultTermination):

    def __init__(self,
                 x_tol=1e-32,
                 cv_tol=1e-6,
                 f_tol=1e-6,
                 period=20,
                 **kwargs) -> None:
        super().__init__(RobustTermination(DesignSpaceTermination(tol=x_tol), period=period),
                         RobustTermination(ConstraintViolationTermination(tol=cv_tol), period=period),
                         RobustTermination(NicheSingleObjectiveSpaceToleranceTermination(tol=f_tol, n_skip=5),
                                           period=period),
                         **kwargs)


# =========================================================================================================
# Selection
# =========================================================================================================


def comp_by_cv_and_clearing_fitness(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV[0] > 0.0 or pop[b].CV[0] > 0.0:
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

    def __init__(self, epsilon, n_max_each_iter=None, norm_by_dim=False) -> None:
        super().__init__(False)
        self.epsilon = epsilon
        self.n_max_each_iter = n_max_each_iter
        self.norm_by_dim = norm_by_dim

    def _do(self, problem, pop, n_survive=None, out=None, **kwargs):
        F = pop.get("F")

        if F.shape[1] != 1:
            raise ValueError("FitnessSurvival can only used for single objective single!")

        # this basically sorts the population by constraint and objective value
        pop = FitnessSurvival().do(problem, pop, n_survive=len(pop))

        # calculate the distance from each individual to another - pre-processing for the clearing
        # NOTE: the distance is normalized by the maximum distance possible
        X = pop.get("X").astype(float)
        D = norm_eucl_dist(problem, X, X)
        if self.norm_by_dim:
            D = D / (problem.n_var ** 0.5)

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
                 norm_by_dim=False,
                 return_all_opt=True,
                 output=NicheOutput(),
                 survival=None,
                 selection=None,
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

        if survival is None:
            survival = EpsilonClearingSurvival(norm_niche_size, n_max_each_iter=None, norm_by_dim=norm_by_dim)

        if selection is None:
            selection = TournamentSelection(comp_by_cv_and_clearing_fitness)

        super().__init__(pop_size=pop_size,
                         selection=selection,
                         survival=survival,
                         output=output,
                         advance_after_initial_infill=True,
                         **kwargs)

        # self.termination = NicheTermination()
        self.termination = DefaultSingleObjectiveTermination()

        # whether with rank one after clearing or just the best should be considered as optimal
        self.return_all_opt = return_all_opt

    def _set_optimum(self, **kwargs):
        if self.return_all_opt:
            self.opt = self.pop[self.pop.get("iter") == 1]
        else:
            super()._set_optimum()


parse_doc_string(NicheGA.__init__)
