import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.factory import get_decomposition
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.util.misc import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.display import disp_multi_objective


# =========================================================================================================
# Implementation
# =========================================================================================================

class MOEAD(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 n_neighbors=20,
                 decomposition='auto',
                 prob_neighbor_mating=0.9,
                 **kwargs):
        """

        Parameters
        ----------
        ref_dirs : {ref_dirs}

        decomposition : {{ 'auto', 'tchebi', 'pbi' }}
            The decomposition approach that should be used. If set to `auto` for two objectives `tchebi` and for more than
            two `pbi` will be used.

        n_neighbors : int
            Number of neighboring reference lines to be used for selection.

        prob_neighbor_mating : float
            Probability of selecting the parents in the neighborhood.


        """

        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating
        self.decomposition = decomposition

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', FloatRandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=1.0, eta=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=None, eta=20))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', None)

        super().__init__(**kwargs)

        self.func_display_attrs = disp_multi_objective

        # initialized when problem is known
        self.ref_dirs = ref_dirs

        if self.ref_dirs.shape[0] < self.n_neighbors:
            print("Setting number of neighbours to population size: %s" % self.ref_dirs.shape[0])
            self.n_neighbors = self.ref_dirs.shape[0]

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]

    def _initialize(self):

        if isinstance(self.decomposition, str):

            # set a string
            decomp = self.decomposition

            # for one or two objectives use tchebi otherwise pbi
            if decomp == 'auto':
                if self.problem.n_obj <= 2:
                    decomp = 'tchebi'
                else:
                    decomp = 'pbi'

            # set the decomposition object
            self._decomposition = get_decomposition(decomp)

        else:
            self._decomposition = self.decomposition

        super()._initialize()
        self.ideal_point = np.min(self.pop.get("F"), axis=0)

    def _next(self):

        # retrieve the current population
        pop = self.pop

        # iterate for each member of the population in random order
        for i in np.random.permutation(len(pop)):

            # all neighbors of this individual and corresponding weights
            N = self.neighbors[i, :]

            if np.random.random() < self.prob_neighbor_mating:
                parents = N[np.random.permutation(self.n_neighbors)][:self.crossover.n_parents]
            else:
                parents = np.random.permutation(self.pop_size)[:self.crossover.n_parents]

            # do recombination and create an offspring
            off = self.crossover.do(self.problem, pop, parents[None, :])
            off = self.mutation.do(self.problem, off)
            off = off[np.random.randint(0, len(off))]

            # repair first in case it is necessary
            if self.repair:
                off = self.repair.do(self.problem, off, algorithm=self)

            # evaluate the offspring
            self.evaluator.eval(self.problem, off)

            # update the ideal point
            self.ideal_point = np.min(np.vstack([self.ideal_point, off.F]), axis=0)

            # calculate the decomposed values for each neighbor
            FV = self._decomposition.do(pop[N].get("F"), weights=self.ref_dirs[N, :], ideal_point=self.ideal_point)
            off_FV = self._decomposition.do(off.F[None, :], weights=self.ref_dirs[N, :], ideal_point=self.ideal_point)

            # get the absolute index in F where offspring is better than the current F (decomposed space)
            I = np.where(off_FV < FV)[0]
            pop[N[I]] = off


# =========================================================================================================
# Interface
# =========================================================================================================

def moead(*args, **kwargs):
    return MOEAD(*args, **kwargs)


parse_doc_string(MOEAD.__init__)
