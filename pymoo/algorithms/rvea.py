import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import has_feasible
from pymoo.util.termination.max_gen import MaximumGenerationTermination


class RVEA(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 fr=0.1,
                 pop_size=None,
                 sampling=FloatRandomSampling(),
                 selection=RandomSelection(),
                 crossover=SimulatedBinaryCrossover(eta=30, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        Parameters
        ----------

        ref_dirs : {ref_dirs}
        fr : float
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
        self.fr = fr

        # you can override the survival if necessary
        survival = kwargs.pop("survival", None)
        if survival is None:
            survival = APDSurvival(ref_dirs)

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         display=display,
                         **kwargs)

    def setup(self, problem, **kwargs):
        super().setup(problem, **kwargs)

        # check whether the n_gen termination is used - otherwise this algorithm can be not run
        if not isinstance(self.termination, MaximumGenerationTermination):
            raise Exception("Please use the n_gen as a termination criterion to execute RVEA!")

    def _next(self):
        super()._next()

        n_gen, n_max_gen = self.n_gen - 1, self.termination.n_max_gen - 1

        # each i-th generation (define by fr and n_max_gen) the reference directions are updated
        if n_gen % np.ceil(n_max_gen * self.fr) == 0:

            # get the objective values
            F = self.pop.get("F")

            # calculate the ideal and nadir point based on the current population
            ideal, nadir = F.min(axis=0), F.max(axis=0)

            # denormalize the reference directions based on the estimations
            ref_dirs = self.ref_dirs * (nadir - ideal)

            self.survival.set_ref_dirs(ref_dirs)

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class APDSurvival(Survival):

    def __init__(self, ref_dirs, alpha=2.0) -> None:
        super().__init__(filter_infeasible=True)
        self.alpha = alpha
        self.V, self.gamma = None, None
        self.set_ref_dirs(ref_dirs)

    def set_ref_dirs(self, ref_dirs):
        self.V = ref_dirs / np.linalg.norm(ref_dirs, axis=1)[:, None]
        self.gamma = np.arccos((- np.sort(-1 * self.V @ self.V.T))[:, 1])
        self.gamma = np.maximum(self.gamma, 1e-64)

    def _do(self, problem, pop, n_survive, algorithm=None, **kwargs):
        problem = algorithm.problem
        n_gen, n_max_gen = algorithm.n_gen - 1, algorithm.termination.n_max_gen
        F = pop.get("F")

        # translate the population
        F = F - F.min(axis=0)

        # the distance to the ideal point
        dist_to_ideal = (F ** 2).sum(axis=1) ** 0.5
        dist_to_ideal[dist_to_ideal < 1e-64] = 1e-64

        # normalize by distance to ideal
        F_prime = F / dist_to_ideal[:, None]

        # calculate for each solution the acute angles to ref dirs
        acute_angle = np.arccos(F_prime @ self.V.T)
        niches = acute_angle.argmin(axis=1)

        # assign to each reference direction the solution
        A = [[] for _ in range(len(self.V))]
        for k, i in enumerate(niches):
            A[i].append(k)

        # all individuals which will be surviving
        survivors = []

        # for each reference direction
        for k in range(len(self.V)):

            # individuals assigned to the niche
            assigned_to_niche = A[k]

            # if niche not empty
            if len(assigned_to_niche) > 0:

                # the angle of niche to nearest neighboring niche
                gamma = self.gamma[k]

                # the angle from the individuals of this niches to the niche itself
                theta = acute_angle[assigned_to_niche, k]

                # the penalty which is applied for the metric
                penalty = problem.n_obj * ((n_gen / n_max_gen) ** self.alpha) * (theta / gamma)

                # calculate the angle-penalized penalized (APD)
                apd = dist_to_ideal[assigned_to_niche] * (1 + penalty)

                # the individual which survives
                survivor = assigned_to_niche[apd.argmin()]

                # set attributes to the individual
                pop[assigned_to_niche].set(theta=theta, apd=apd, niche=k, best=False)
                pop[survivor].set("best", True)

                # select the one with smallest APD value
                survivors.append(survivor)

        return pop[survivors]


parse_doc_string(RVEA.__init__)
