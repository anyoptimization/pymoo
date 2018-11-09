import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.real_differental_evolution_crossover import DifferentialEvolutionCrossover
from pymoo.operators.default_operators import set_default_if_none
from pymoo.operators.mutation.real_differential_evoluation_mutation import DifferentialEvolutionMutation
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.util.display import disp_single_objective


class DifferentialEvolution(GeneticAlgorithm):
    def __init__(self,
                 **kwargs):
        set_default_if_none("real", kwargs)
        super().__init__(**kwargs)
        self.selection = RandomSelection()
        self.crossover = DifferentialEvolutionCrossover(weight=0.75)
        self.mutation = DifferentialEvolutionMutation("binomial", 0.1)
        self.func_display_attrs = disp_single_objective

    def _next(self, pop):

        # create offsprings and add it to the data of the algorithm
        P = self.selection.do(pop, self.pop_size, self.crossover.n_parents)
        off = self.crossover.do(self.problem, pop, P)
        self.data = {**self.data, "off": off}

        # do the mutation by using the offsprings
        off = self.mutation.do(self.problem, pop, D=self.data)

        # evaluate the results
        self.evaluator.eval(self.problem, off, D=self.data)

        # replace whenever offspring is better than population member
        for i in range(len(pop)):
            if off[i].F < pop[i].F:
                pop[i] = off[i]

        return pop
