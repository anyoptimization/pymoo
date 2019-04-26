import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.differental_evolution_crossover import DifferentialEvolutionCrossover
from pymoo.operators.crossover.exponential_crossover import ExponentialCrossover
from pymoo.operators.crossover.uniform_crossover import UniformCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.repair.bounds_back_repair import BoundsBackRepair
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.rand import random
from pymoo.util.display import disp_single_objective
from pymoo.util.misc import parameter_less


# =========================================================================================================
# Implementation
# =========================================================================================================


class DifferentialEvolution(GeneticAlgorithm):
    def __init__(self,
                 variant,
                 CR,
                 F,
                 dither,
                 jitter,
                 **kwargs):

        _, self.var_selection, self.var_n, self.var_mutation, = variant.split("/")

        set_if_none(kwargs, 'pop_size', 200)
        set_if_none(kwargs, 'sampling', LatinHypercubeSampling(criterion="maxmin", iterations=100))
        set_if_none(kwargs, 'crossover', DifferentialEvolutionCrossover(weight=F, dither=dither, jitter=jitter))
        set_if_none(kwargs, 'selection', RandomSelection())

        if self.var_mutation == "exp":
            set_if_none(kwargs, 'mutation', ExponentialCrossover(CR))
        elif self.var_mutation == "bin":
            set_if_none(kwargs, 'mutation', UniformCrossover(CR))

        set_if_none(kwargs, 'survival', None)
        super().__init__(**kwargs)

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

        # do the first crossover which is the actual DE operation
        self.off = self.crossover.do(self.problem, pop, P, algorithm=self)

        # then do the mutation (which is actually )
        _pop = self.off.new().merge(self.pop).merge(self.off)
        _P = np.column_stack([np.arange(len(pop)), np.arange(len(pop)) + len(pop)])
        self.off = self.mutation.do(self.problem, _pop, _P, algorithm=self)[:len(self.pop)]

        # bounds back if something is out of bounds
        self.off = BoundsBackRepair().do( self.problem, self.off)

        # evaluate the results
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        _F, _CV, _feasible = self.off.get("F", "CV", "feasible")
        _F = parameter_less(_F, _CV)

        # find the individuals which are indeed better
        is_better = np.where((_F <= F)[:, 0])[0]

        # replace the individuals in the population
        pop[is_better] = self.off[is_better]

        return pop


# =========================================================================================================
# Interface
# =========================================================================================================


def de(
        pop_size=100,
        sampling=LatinHypercubeSampling(iterations=100, criterion="maxmin"),
        variant="DE/rand/1/bin",
        CR=0.5,
        F=0.3,
        dither="vector",
        jitter=False,
        **kwargs):
    """

    Parameters
    ----------

    pop_size : {pop_size}

    sampling : {sampling}

    variant : {{DE/(rand|best)/1/(bin/exp)}}
        The different variants of DE to be used. DE/x/y/z where x how to select individuals to be pertubed,
        y the number of difference vector to be used and z the crossover type. One of the most common variant
        is DE/rand/1/bin.

    F : float
        The weight to be used during the crossover.

    CR : float
        The probability the individual exchanges variable values from the donor vector.

    dither : {{'no', 'scalar', 'vector'}}
        One strategy to introduce adaptive weights (F) during one run. The option allows
        the same dither to be used in one iteration ('scalar') or a different one for
        each individual ('vector).

    jitter : bool
        Another strategy for adaptive weights (F). Here, only a very small value is added or
        substracted to the weight used for the crossover for each individual.


    Returns
    -------
    de : :class:`~pymoo.model.algorithm.Algorithm`
        Returns an DifferentialEvolution algorithm object.

    """

    _, _selection, _n, _mutation, = variant.split("/")

    return DifferentialEvolution(
        variant,
        CR,
        F,
        dither,
        jitter,
        pop_size=pop_size,
        sampling=sampling,
        **kwargs)


parse_doc_string(de)
