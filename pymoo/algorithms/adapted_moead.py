from scipy.spatial.distance import cdist
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.factory import get_decomposition, get_performance_indicator
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import set_if_none

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# =========================================================================================================
# Implementation
# =========================================================================================================

class MOEAD(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 n_neighbors=20,
                 decomposition='auto',
                 prob_neighbor_mating=0.9,
                 display=MultiObjectiveDisplay(),
                 current_execution_number=0,
                 save_dir='',
                 save_data=True,
                 **kwargs):
        """

        MOEAD Algorithm.

        Parameters
        ----------
        ref_dirs
        n_neighbors
        decomposition
        prob_neighbor_mating
        display
        kwargs
        """

        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating
        self.decomposition = decomposition
        self.current_execution_number = current_execution_number
        self.save_dir = save_dir
        self.save_data = save_data
        self.hvs = []
        self.igds = []

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', FloatRandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=1.0, eta=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=None, eta=20))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', None)

        super().__init__(display=display, **kwargs)

        # initialized when problem is known
        self.ref_dirs = ref_dirs

        if self.ref_dirs.shape[0] < self.n_neighbors:
            print("Setting number of neighbours to population size: %s" % self.ref_dirs.shape[0])
            self.n_neighbors = self.ref_dirs.shape[0]

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]
        self.current_generation = 0

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

        self.hv = get_performance_indicator("hv", ref_point=np.array([1.2]*self.problem.n_obj))
        self.igd_plus = get_performance_indicator("igd+", self.problem.pareto_front(ref_dirs=self.ref_dirs))
        self.create_result_folders()
        

    def _next(self):
        repair, crossover, mutation = self.repair, self.mating.crossover, self.mating.mutation

        # retrieve the current population
        pop = self.pop

        print('Current generation:', self.current_generation)
        current_hv = self.get_hypervolume(pop)
        current_igd = self.get_igd(pop)
        self.hvs.append(current_hv)
        self.igds.append(current_igd)

        print('Metrics HV {} IDG+ {}'.format(current_hv, current_igd))

        if self.save_data:
            self.save_current_iteration_files(pop)

        # iterate for each member of the population in random order
        for i in np.random.permutation(len(pop)):

            # all neighbors of this individual and corresponding weights
            N = self.neighbors[i, :]

            if np.random.random() < self.prob_neighbor_mating:
                parents = N[np.random.permutation(self.n_neighbors)][:crossover.n_parents]
            else:
                parents = np.random.permutation(self.pop_size)[:crossover.n_parents]

            # do recombination and create an offspring
            off = crossover.do(self.problem, pop, parents[None, :])
            off = mutation.do(self.problem, off)
            off = off[np.random.randint(0, len(off))]

            # repair first in case it is necessary - disabled if instance of NoRepair
            off = repair.do(self.problem, off, algorithm=self)

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

    def _finalize(self):        
        if self.save_data:
            self.save_algorithm_data('hv_convergence.txt', self.hvs)
            self.save_algorithm_data('igd_convergence.txt', self.igds)

    def get_hypervolume(self, population):
        return self.hv.calc(population.get('F'))
    
    def get_igd(self, population):
        return self.igd_plus.calc(population.get('F'))

    def save_current_iteration_files(self, population):
        variables = [individual.get('X') for individual in population]
        objectives = [individual.get('F') for individual in population]
        self.save_algorithm_data('variables_{}.txt'.format(self.current_generation), variables)
        self.save_algorithm_data('objectives_{}.txt'.format(self.current_generation), objectives)
        
    def save_algorithm_data(self, file_name, data_list):
        with open(os.path.join(self.full_path, file_name),'w') as file:
            for data in data_list:
                file.write(str(data) + '\n')

    def create_result_folders(self):
        folder = 'Execution {}'.format(self.current_execution_number)
        self.full_path = os.path.join(self.save_dir, folder)
        
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)
            print('Execution folder created!')
        else:
            print('Folder already exists!')
