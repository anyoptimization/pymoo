import os
import time
import numpy as np

from pymoo.model.algorithm import Algorithm
from pymoo.model.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.model.individual import Individual
from pymoo.model.initialization import Initialization
from pymoo.model.mating import Mating
from pymoo.model.population import Population
from pymoo.model.repair import NoRepair
from pymoo.factory import get_performance_indicator


class AdaptedGeneticAlgorithm(Algorithm):

    def __init__(self,
                 ref_dirs,
                 pop_size=None,
                 sampling=None,
                 selection=None,
                 crossover=None,
                 mutation=None,
                 survival=None,
                 n_offsprings=None,
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 repair=None,
                 mating=None,
                 individual=Individual(),
                 min_infeas_pop_size=0,
                 current_execution_number=0,
                 save_dir='',
                 save_data=True,
                 **kwargs
                 ):

        super().__init__(**kwargs)
        self.start = time.time()
        # the population size used
        self.pop_size = pop_size

        # minimum number of individuals surviving despite being infeasible - by default disabled
        self.min_infeas_pop_size = min_infeas_pop_size

        # the survival for the genetic algorithm
        self.survival = survival

        # number of offsprings to generate through recombination
        self.n_offsprings = n_offsprings

        # if the number of offspring is not set - equal to population size
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        # the object to be used to represent an individual - either individual or derived class
        self.individual = individual

        # set the duplicate detection class - a boolean value chooses the default duplicate detection
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        # simply set the no repair object if it is None
        self.repair = repair if repair is not None else NoRepair()

        self.initialization = Initialization(sampling,
                                             individual=individual,
                                             repair=self.repair,
                                             eliminate_duplicates=self.eliminate_duplicates)

        if mating is None:
            mating = Mating(selection,
                            crossover,
                            mutation,
                            repair=self.repair,
                            eliminate_duplicates=self.eliminate_duplicates,
                            n_max_iterations=100)
        self.mating = mating

        # other run specific data updated whenever solve is called - to share them in all algorithms
        self.n_gen = None
        self.pop = None
        self.off = None

        self.ref_dirs = ref_dirs
        self.current_execution_number = current_execution_number
        self.save_dir = save_dir
        self.save_data = save_data
        self.hvs = []
        self.igds = []
        self.current_generation = 0


    def _initialize(self):

        # create the initial population
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        pop.set("n_gen", self.n_gen)

        # then evaluate using the objective function
        self.evaluator.eval(self.problem, pop, algorithm=self)

        # that call is a dummy survival to set attributes that are necessary for the mating selection
        if self.survival:
            pop = self.survival.do(self.problem, pop, len(pop), algorithm=self,
                                   n_min_infeas_survive=self.min_infeas_pop_size)

        self.pop, self.off = pop, pop

        self.hv = get_performance_indicator("hv", ref_point=np.array([1.2]*self.problem.n_obj))
        self.igd_plus = get_performance_indicator("igd+", self.problem.pareto_front(ref_dirs=self.ref_dirs))
        self.create_result_folders()
        
    def _next(self):

        # do the mating using the current population
        self.off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        self.off.set("n_gen", self.n_gen)

        if self.save_data:
            self.save_current_iteration_files(self.pop)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(self.off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(self.off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        # evaluate the offspring
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        # merge the offsprings with the current population
        self.pop = Population.merge(self.pop, self.off)

        # the do survival selection
        if self.survival:
            self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self,
                                        n_min_infeas_survive=self.min_infeas_pop_size)
        #current_hv = self.get_hypervolume(self.pop)
        current_igd = self.get_igd(self.pop)
        #self.hvs.append(current_hv)
        self.igds.append(current_igd)

        self.current_generation += 1
        print(self.current_generation)

    def _finalize(self):
        for individual in self.pop:
            individual.F = self.problem.evaluate(individual.get('X'))
        
        if self.save_data:
            self.save_algorithm_data('hv_convergence.txt', self.hvs)
            self.save_algorithm_data('igd_convergence.txt', self.igds)
            self.save_algorithm_data('time.txt', [time.time() - self.start])
    
    def _get_hypervolume(self, population):
        return self.hv.calc(population.get('F'))
    
    def _get_igd(self, population):
        return self.igd_plus.calc(population.get('F'))
    
    def get_hypervolume(self, population):
        return self.hv.calc(population.get('F'))
    
    def get_igd(self, population):
        return self.igd_plus.calc(population.get('F'))

    def save_current_iteration_files(self, population):
        # variables = [individual.get('X') for individual in population]
        objectives = [individual.get('F') for individual in population]
        # self.save_algorithm_data('variables_{}.txt'.format(self.current_generation), variables)
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
