import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.differental_evolution_crossover import DifferentialEvolutionCrossover
from pymoo.operators.default_operators import set_default_if_none
from pymoo.operators.mutation.differential_evoluation_mutation import DifferentialEvolutionMutation
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.util.display import disp_single_objective


class DifferentialEvolution(GeneticAlgorithm):
    def __init__(self,
                 variant="DE/rand/1/exp",
                 CR=0.1,
                 F=0.75,
                 **kwargs):
        set_default_if_none("real", kwargs)
        super().__init__(**kwargs)
        self.selection = RandomSelection()

        self.crossover = DifferentialEvolutionCrossover(weight=F)

        _, self.var_selection, self.var_n, self.var_mutation, = variant.split("/")

        self.mutation = DifferentialEvolutionMutation(self.var_mutation, CR)
        self.func_display_attrs = disp_single_objective

    def _next(self, pop):

        # create offsprings and add it to the data of the algorithm
        if self.var_selection == "rand":
            P = self.selection.do(pop, self.pop_size, self.crossover.n_parents)
        elif self.var_selection == "best":
            P = self.selection.do(pop, self.pop_size, self.crossover.n_parents - 1)
            best = np.argmin(pop.get("F")[:, 0])
            P = np.hstack([np.full(len(pop), best)[:, None], P])
        else:
            raise Exception("Unknown selection: %s" % self.var_selection)

        self.off = self.crossover.do(self.problem, pop, P)

        # do the mutation by using the offsprings
        self.off = self.mutation.do(self.problem, pop, algorithm=self)

        # evaluate the results
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        # replace whenever offspring is better than population member
        for i in range(len(pop)):
            if self.off[i].F < pop[i].F:
                pop[i] = self.off[i]

        return pop
