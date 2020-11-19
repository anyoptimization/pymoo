import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.algorithms.rvea import RVEA
from pymoo.factory import get_problem, get_reference_directions
from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.selection.tournament_selection import compare, TournamentSelection
from pymoo.optimize import minimize
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import has_feasible, vectorized_cdist
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination
from pymoo.util.termination.max_gen import MaximumGenerationTermination
from pymoo.visualization.scatter import Scatter


def binary_tournament(pop, P, algorithm, **kwargs):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):

        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            S[i] = compare(a, pop[a].get("rank"), b, pop[b].get("rank"),
                           method='smaller_is_better', return_random_if_equal=True)

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, pop[a].get("apd"), b, pop[b].get("apd"),
                               method='smaller_is_better', return_random_if_equal=True)

    return S[:, None].astype(np.int, copy=False)


class ModifiedRVEA(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 alpha=2.0,
                 adapt_freq=0.1,
                 pop_size=None,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(binary_tournament),
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
            survival = ModifiedAPDSurvival(ref_dirs, alpha=alpha)

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

        # if maximum functions termination convert it to generations
        if isinstance(self.termination, MaximumFunctionCallTermination):
            n_gen = np.ceil((self.termination.n_max_evals - self.pop_size) / self.n_offsprings)
            self.termination = MaximumGenerationTermination(n_gen)

        # check whether the n_gen termination is used - otherwise this algorithm can be not run
        if not isinstance(self.termination, MaximumGenerationTermination):
            raise Exception("Please use the n_gen or n_eval as a termination criterion to execute RVEA!")

    def _next(self):
        super()._next()

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


class ModifiedAPDSurvival(Survival):

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
        self.V = calc_V(self.ref_dirs * (self.nadir - self.ideal))
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

        # do the non-dominated sorting
        fronts = NonDominatedSorting().do(F)

        for rank, front in enumerate(fronts):

            # for each reference direction
            for k in range(len(self.V)):

                # individuals assigned to the niche
                assigned_to_niche = [niche for niche in niches_to_ind[k] if niche in front]

                # if niche not empty
                if len(assigned_to_niche) > 0:
                    # the angle of niche to nearest neighboring niche
                    gamma = self.gamma[k]

                    # the angle from the individuals of this niches to the niche itself
                    theta = acute_angle[assigned_to_niche, k]

                    # the penalty which is applied for the metric
                    M = problem.n_obj if problem.n_obj > 2 else 1.0
                    penalty = M * ((n_gen / n_max_gen) ** self.alpha) * (theta / gamma)

                    # calculate the angle-penalized penalized (APD)
                    apd = dist_to_ideal[assigned_to_niche] * (1 + penalty)

                    # the individual which survives
                    closest = assigned_to_niche[apd.argmin()]

                    # set attributes to the individual
                    pop[assigned_to_niche].set(theta=theta, apd=apd, niche=k, closest=False, rank=rank)
                    pop[closest].set("closest", True)

                    if k in self.extreme_ref_dirs:
                        on_boundary = np.argmin(theta)
                        if on_boundary != closest:
                            pop[on_boundary].set("closest", True)

            if rank == 1:
                nds = pop[front]
                self.nadir = nds[nds.get("closest")].get("F").max(axis=0)

        I = np.lexsort([pop.get("apd"), pop.get("rank")])

        self.niches = niches_to_ind

        return pop[I][:n_survive]


if __name__ == "__main__":
    problem = get_problem("zdt1")

    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=99)

    algorithm = ModifiedRVEA(ref_dirs, adapt_freq=2.0)

    pf = problem.pareto_front()

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 200),
                   seed=1,
                   pf=pf,
                   verbose=True)

    plot = Scatter()
    plot.add(pf, plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, color="red")
    plot.show()
