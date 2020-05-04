import matplotlib.pyplot as plt
import numpy as np
from pyrecorder.recorders.file import File
from pyrecorder.video import Video

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.algorithms.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.docs import parse_doc_string
from pymoo.model.mating import Mating
from pymoo.model.population import Population
from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.operators.selection.tournament_selection import TournamentSelection, compare
from pymoo.optimize import minimize
from pymoo.problems.single.multimodal import MultiModalSimple1, curve, MultiModalSimple2
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.misc import vectorized_cdist, norm_eucl_dist
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from pymoo.visualization.scatter import Scatter


# =========================================================================================================
# Implementation
# =========================================================================================================


def comp_by_rank(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]
        S[i] = compare(a, pop[a].get("rank"), b, pop[b].get("rank"), method='smaller_is_better',
                       return_random_if_equal=True)

    return S[:, None].astype(np.int)


class MMGA(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=RandomSelection(),
                 crossover=SimulatedBinaryCrossover(prob=0.9, eta=3),
                 mutation=PolynomialMutation(prob=None, eta=5),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=SingleObjectiveDisplay(),
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

        """

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=NichingSurvival(),
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         display=display,
                         **kwargs)

        # self.mating = NeighborBiasedMating(selection,
        #                                    crossover,
        #                                    mutation,
        #                                    repair=self.mating.repair,
        #                                    eliminate_duplicates=self.mating.eliminate_duplicates,
        #                                    n_max_iterations=self.mating.n_max_iterations)

        self.default_termination = SingleObjectiveDefaultTermination()


class NichingSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(True)

    def _do(self, problem, pop, n_survive, out=None, algorithm=None, **kwargs):
        X, F = pop.get("X", "F")
        if F.shape[1] != 1:
            raise ValueError("FitnessSurvival can only used for single objective single!")

        n_neighbors = 5

        # calculate the normalized euclidean distances from each solution to another
        D = norm_eucl_dist(problem, X, X, fill_diag_with_inf=True)

        # set the neighborhood for each individual
        for k, individual in enumerate(pop):

            # the neighbors in the current population
            neighbors = pop[D[k].argsort()[:n_neighbors]]

            # get the neighbors of the current individual and merge
            N = individual.get("neighbors")
            if N is not None:
                rec = []
                h = set()
                for n in N:
                    for entry in n.get("neighbors"):
                        if entry not in h:
                            rec.append(entry)
                            h.add(entry)

                neighbors = Population.merge(neighbors, rec)

            # keep only the closest solutions to the individual
            _D = norm_eucl_dist(problem, individual.X[None, :], neighbors.get("X"))[0]

            # find only the closest neighbors
            closest = _D.argsort()[:n_neighbors]

            individual.set("crowding", _D[closest].mean())
            individual.set("neighbors", neighbors[closest])


        best = F[:, 0].argmin()
        print(F[best], pop[best].get("crowding"))

        # plt.scatter(F[:, 0], pop.get("crowding"))
        # plt.show()

        pop.set("_F", pop.get("F"))
        pop.set("F", np.column_stack([F, -pop.get("crowding")]))
        pop = RankAndCrowdingSurvival().do(problem, pop, n_survive)
        pop.set("F", pop.get("_F"))

        return pop


class NeighborBiasedMating(Mating):

    def __init__(self, selection, crossover, mutation, bias=0.7, **kwargs):
        super().__init__(selection, crossover, mutation, **kwargs)
        self.bias = bias

    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):
        rnd = np.random.random(n_offsprings)
        n_neighbors = (rnd <= self.bias).sum()

        other = super()._do(problem, pop, n_offsprings - n_neighbors, parents, **kwargs)

        N = []

        cand = TournamentSelection(comp_by_rank).do(pop, n_neighbors, n_parents=1)[:, 0]
        for k in cand:
            N.append(pop[k])

            n_cand_neighbors = pop[k].get("neighbors")
            rnd = np.random.permutation(len(n_cand_neighbors))[:self.crossover.n_parents - 1]
            [N.append(e) for e in n_cand_neighbors[rnd]]

        parents = np.reshape(np.arange(len(N)), (-1, self.crossover.n_parents))
        N = Population.create(*N)

        bias = super()._do(problem, N, n_neighbors, parents, **kwargs)

        return Population.merge(bias, other)


parse_doc_string(MMGA.__init__)

if __name__ == '__main__':
    problem = MultiModalSimple2()

    algorithm = MMGA(
        pop_size=20,
        eliminate_duplicates=True)

    ret = minimize(problem,
                   algorithm,
                   termination=('n_gen', 100),
                   seed=1,
                   save_history=True,
                   verbose=False)


    def plot(algorithm):
        pop = algorithm.pop
        sc = Scatter(title=algorithm.n_gen)
        sc.add(curve(algorithm.problem), plot_type="line", color="black")
        sc.add(np.column_stack([pop.get("X"), pop.get("F")]), color="red")
        sc.do()


    plot(ret.algorithm)
    plt.show()

    with Video(File("mm.mp4")) as vid:
        for entry in ret.history:
            plot(entry)
            vid.record()
