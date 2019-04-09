import numpy as np
from pymoo.rand import random

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.differental_evolution_crossover import DifferentialEvolutionCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.differential_evoluation_mutation import DifferentialEvolutionMutation
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.util.display import disp_single_objective
from pymoo.util.misc import parameter_less


# =========================================================================================================
# Implementation
# =========================================================================================================


class DifferentialEvolution(GeneticAlgorithm):
    def __init__(self,
                 variant="DE/rand+best/1/bin",
                 CR=0.5,
                 F=0.75,
                 n_replace=None,
                 **kwargs):

        _, self.var_selection, self.var_n, self.var_mutation, = variant.split("/")

        set_if_none(kwargs, 'pop_size', 200)
        set_if_none(kwargs, 'sampling', LatinHypercubeSampling(criterion="maxmin", iterations=100))
        set_if_none(kwargs, 'crossover', DifferentialEvolutionCrossover(weight=F))
        set_if_none(kwargs, 'selection', RandomSelection())
        set_if_none(kwargs, 'mutation', DifferentialEvolutionMutation(self.var_mutation, CR))
        set_if_none(kwargs, 'survival', None)
        super().__init__(**kwargs)

        self.n_replace = n_replace
        self.func_display_attrs = disp_single_objective

    def _next(self, pop):

        # get the vectors from the population
        F, CV, feasible = pop.get("F", "CV", "feasible")
        F = parameter_less(F, CV)

        # create offsprings and add it to the data of the algorithm
        if self.var_selection == "rand":
            P = self.selection.do(pop, self.pop_size, self.crossover.n_parents)

        elif self.var_selection == "best":
            best = np.argmin(F[:, 0])
            P = self.selection.do(pop, self.pop_size, self.crossover.n_parents - 1)
            P = np.column_stack([np.full(len(pop), best), P])

        elif self.var_selection == "rand+best":
            best = np.argmin(F[:, 0])
            P = self.selection.do(pop, self.pop_size, self.crossover.n_parents)
            use_best = random.random(len(pop)) < 0.3
            P[use_best, 0] = best

        else:
            raise Exception("Unknown selection: %s" % self.var_selection)

        self.off = self.crossover.do(self.problem, pop, P)

        # do the mutation by using the offsprings
        self.off = self.mutation.do(self.problem, self.off, algorithm=self)

        # bring back to bounds if violated through crossover - bounce back strategy
        X = self.off.get("X")
        xl = np.repeat(self.problem.xl[None, :], X.shape[0], axis=0)
        xu = np.repeat(self.problem.xu[None, :], X.shape[0], axis=0)

        # otherwise bounds back into the feasible space
        X[X < xl] = (xl + (xl - X))[X < xl]
        X[X > xu] = (xu - (X - xu))[X > xu]
        self.off.set("X", X)

        # evaluate the results
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        _F, _CV, _feasible = self.off.get("F", "CV", "feasible")
        _F = parameter_less(_F, _CV)

        # find the individuals which are indeed better
        is_better = np.where((_F <= F)[:, 0])[0]

        # truncate the replacements if desired
        if self.n_replace is not None and self.n_replace < len(is_better):
            is_better = is_better[random.perm(len(is_better))[:self.n_replace]]

        # replace the individuals in the population
        pop[is_better] = self.off[is_better]

        return pop

# =========================================================================================================
# Interface
# =========================================================================================================
