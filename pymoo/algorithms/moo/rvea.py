import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.misc import has_feasible, vectorized_cdist


class RVEA(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 alpha=2.0,
                 adapt_freq=0.1,
                 pop_size=None,
                 sampling=FloatRandomSampling(),
                 selection=RandomSelection(),
                 crossover=SBX(eta=30, prob=1.0),
                 mutation=PM(eta=20),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 output=MultiObjectiveOutput(),
                 **kwargs):
        """

        Parameters
        ----------

        ref_dirs : {ref_dirs}
        adapt_freq : float
            Defines the ratio of generation when the reference directions are updated.
        pop_size : int (default = None)
            By default the population size is set to None which means that it will be equal to the number of reference
            line. However, if desired this can be overwritten by providing a positive number.
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """

        # set reference directions and pop_size
        self.ref_dirs = ref_dirs
        if self.ref_dirs is not None:
            if pop_size is None:
                pop_size = len(self.ref_dirs)

        # the fraction of n_max_gen when the the reference directions are adapted
        self.adapt_freq = adapt_freq

        # you can override the survival if necessary
        survival = kwargs.pop("survival", None)
        if survival is None:
            survival = APDSurvival(ref_dirs, alpha=alpha)

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         output=output,
                         **kwargs)

    def _setup(self, problem, **kwargs):

        # if maximum functions termination convert it to generations
        if isinstance(self.termination, MaximumFunctionCallTermination):
            n_gen = np.ceil((self.termination.n_max_evals - self.pop_size) / self.n_offsprings)
            self.termination = MaximumGenerationTermination(n_gen)

        # check whether the n_gen termination is used - otherwise this algorithm can be not run
        if not isinstance(self.termination, MaximumGenerationTermination):
            raise Exception("Please use the n_gen or n_eval as a termination criterion to run RVEA!")

    def _advance(self, **kwargs):
        super()._advance(**kwargs)

        # get the  current generation and maximum of generations
        n_gen, n_max_gen = self.n_gen, self.termination.n_max_gen

        # each i-th generation (define by fr and n_max_gen) the reference directions are updated
        if self.adapt_freq is not None and n_gen % np.ceil(n_max_gen * self.adapt_freq) == 0:
            self.survival.adapt()

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------

def calc_gamma(V):
    gamma = np.arccos((- np.sort(-1 * V @ V.T))[:, 1])
    gamma = np.maximum(gamma, 1e-64)
    return gamma


def calc_V(ref_dirs):
    return ref_dirs / np.linalg.norm(ref_dirs, axis=1)[:, None]


class APDSurvival(Survival):

    def __init__(self, ref_dirs, alpha=2.0) -> None:
        super().__init__(filter_infeasible=True)
        n_dim = ref_dirs.shape[1]

        self.alpha = alpha
        self.niches = None
        self.V, self.gamma = None, None
        self.ideal, self.nadir = np.full(n_dim, np.inf), None

        self.ref_dirs = ref_dirs
        self.extreme_ref_dirs = np.where(np.any(vectorized_cdist(self.ref_dirs, np.eye(n_dim)) == 0, axis=1))[0]

        self.V = calc_V(self.ref_dirs)
        self.gamma = calc_gamma(self.V)

    def adapt(self):
        if self.nadir is not None:
            self.V = calc_V(calc_V(self.ref_dirs) * (self.nadir - self.ideal))
            self.gamma = calc_gamma(self.V)

    def _do(self, problem, pop, n_survive, algorithm=None, n_gen=None, n_max_gen=None, **kwargs):

        if n_gen is None:
            n_gen = algorithm.n_gen - 1
        if n_max_gen is None:
            n_max_gen = algorithm.termination.n_max_gen

        # get the objective space values
        F = pop.get("F")

        # store the ideal and nadir point estimation for adapt - (and ideal for transformation)
        self.ideal = np.minimum(F.min(axis=0), self.ideal)

        # translate the population to make the ideal point the origin
        F = F - self.ideal

        # the distance to the ideal point
        dist_to_ideal = np.linalg.norm(F, axis=1)
        dist_to_ideal[dist_to_ideal < 1e-64] = 1e-64

        # normalize by distance to ideal
        F_prime = F / dist_to_ideal[:, None]

        # calculate for each solution the acute angles to ref dirs
        acute_angle = np.arccos(F_prime @ self.V.T)
        niches = acute_angle.argmin(axis=1)

        # assign to each reference direction the solution
        niches_to_ind = [[] for _ in range(len(self.V))]
        for k, i in enumerate(niches):
            niches_to_ind[i].append(k)

        # all individuals which will be surviving
        survivors = []

        # for each reference direction
        for k in range(len(self.V)):

            # individuals assigned to the niche
            assigned_to_niche = niches_to_ind[k]

            # if niche not empty
            if len(assigned_to_niche) > 0:

                # the angle of niche to nearest neighboring niche
                gamma = self.gamma[k]

                # the angle from the individuals of this niches to the niche itself
                theta = acute_angle[assigned_to_niche, k]

                # the penalty which is applied for the metric
                M = problem.n_obj if problem.n_obj > 2.0 else 1.0
                penalty = M * ((n_gen / n_max_gen) ** self.alpha) * (theta / gamma)

                # calculate the angle-penalized penalized (APD)
                apd = dist_to_ideal[assigned_to_niche] * (1 + penalty)

                # the individual which survives
                survivor = assigned_to_niche[apd.argmin()]

                # set attributes to the individual
                pop[assigned_to_niche].set(theta=theta, apd=apd, niche=k, opt=False)
                pop[survivor].set("opt", True)

                # select the one with smallest APD value
                survivors.append(survivor)

        ret = pop[survivors]
        self.niches = niches_to_ind
        self.nadir = ret.get("F").max(axis=0)

        return ret


parse_doc_string(RVEA.__init__)
