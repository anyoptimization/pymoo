import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.differental_evolution_crossover import DifferentialEvolutionCrossover
from pymoo.operators.default_operators import set_default_if_none, set_if_none
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.util.display import disp_single_objective


class DifferentialEvolution(GeneticAlgorithm):
    def __init__(self,
                 **kwargs):
        set_default_if_none("real", kwargs)
        super().__init__(**kwargs)


        self.crossover = DifferentialEvolutionCrossover(prob=0.5, weight=0.75, variant="DE/best/1", method="binomial")
        self.func_display_attrs = disp_single_objective

    def _next(self, pop):

        # all neighbors shuffled (excluding the individual itself)
        P = RandomSelection().do(pop, self.pop_size, self.crossover.n_parents - 1)

        origin = None
        if "rand" in self.crossover.variant:
            origin = np.arange(self.pop_size)
        elif "best" in self.crossover.variant:
            origin = np.full(self.pop_size, np.argmin(pop.F))

        P = np.concatenate([origin[:, None], P], axis=1)

        # do recombination and create an offspring
        X = self.crossover.do(self.problem, pop.X[P, :])
        F, _ = self.evaluator.eval(self.problem, X)

        # replace whenever offspring is better than population member
        off_is_better = np.where(F < pop.F)[0]
        pop.F[off_is_better, :] = F[off_is_better, :]
        pop.X[off_is_better, :] = X[off_is_better, :]