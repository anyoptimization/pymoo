import math

import numpy as np

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.algorithms.so_genetic_algorithm import FitnessSurvival
from pymoo.docs import parse_doc_string
from pymoo.model.infill import InfillCriterion
from pymoo.model.population import Population
from pymoo.model.replacement import ImprovementReplacement
from pymoo.operators.repair.to_bound import ToBoundOutOfBoundsRepair
from pymoo.operators.sampling.latin_hypercube_sampling import LHS
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.termination.default import SingleObjectiveDefaultTermination


# =========================================================================================================
# Levy
# =========================================================================================================


class MantegnasAlgorithm:

    def __init__(self, beta) -> None:
        """
        This algorithm can be used to sample to levy flight.

        Parameters
        ----------
        beta : float
            The parameter of the levy distribution

        """
        super().__init__()

        # calculate the constant that we can actually sample form the normal distribution
        a = math.gamma(1. + beta) * math.sin(math.pi * beta / 2.)
        b = beta * math.gamma((1. + beta) / 2.) * 2 ** ((beta - 1.) / 2)
        self.s_u = (a / b) ** (1. / (2 * beta))
        self.beta = beta
        self.s_v = 1

    def do(self, size=None):
        u = np.random.normal(0, self.s_u, size)
        v = np.abs(np.random.normal(0, self.s_v, size)) ** (1. / self.beta)
        return u / v


class LevyFlights(InfillCriterion):

    def __init__(self, alpha, beta, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.levy = MantegnasAlgorithm(beta)

    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):
        if parents is None:
            raise Exception("For levy flights please provide the parents!")

        X, F = pop.get("X", "F")
        xl, xu = problem.bounds()

        a, b, c = parents.T

        # the direction to be used for improvement
        direction = (X[b] - X[c])

        # get random levy values to be used for the step size
        levy = self.levy.do(size=(len(parents), 1))
        # levy = np.random.normal(0, 1, size=(len(parents), problem.n_var))

        _X = X[a] + (xu - xl) / self.alpha * levy * direction

        _X = ToBoundOutOfBoundsRepair().do(problem, _X)
        # _X = InversePenaltyOutOfBoundsRepair().do(problem, _X, P=X[a])

        return Population.new(X=_X, index=a)


# =========================================================================================================
# Implementation
# =========================================================================================================


class CuckooSearch(GeneticAlgorithm):

    def __init__(self,
                 pop_size=25,
                 n_offsprings=None,
                 sampling=LHS(),
                 termination=SingleObjectiveDefaultTermination(),
                 display=SingleObjectiveDisplay(),
                 beta=1.5,
                 alpha=0.01,
                 pa=0.1,
                 **kwargs):
        """

        Parameters
        ----------

        sampling : {sampling}

        termination : {termination}

        pop_size : int
         The number of nests to be used

        beta : float
            The input parameter of the Mantegna's Algorithm to simulate
            sampling on Levy Distribution

        alpha : float
            The step size scaling factor and is usually 0.01.

        pa : float
            The switch probability, pa fraction of the nests will be abandoned on every iteration
        """
        mating = kwargs.get("mating")
        if mating is None:
            mating = LevyFlights(alpha, beta)

        super().__init__(pop_size=pop_size, n_offsprings=n_offsprings, sampling=sampling, mating=mating,
                         termination=termination, display=display, **kwargs)

        self.pa = pa

    def _next(self):
        pop, n_offsprings = self.pop, self.n_offsprings

        best = FitnessSurvival().do(self.problem, pop, 1, return_indices=True)[0]

        perm = lambda: np.random.permutation(len(pop))

        # randomly select the parents and to be used for mating
        a = perm()[:n_offsprings]
        # a = np.repeat(best, n_offsprings)

        # b = perm()[:n_offsprings]
        # b[np.random.random(len(b)) < 0.25] = best

        # a = np.repeat(best, n_offsprings)
        b = np.repeat(best, n_offsprings)
        # b = np.random.permutation(len(pop))[:n_offsprings]
        c = perm()[:n_offsprings]

        P = np.column_stack([a, b, c])

        # do the levy flight mating and evaluate the result offsprings
        off = self.mating.do(self.problem, pop, len(pop), parents=P, algorithm=self)
        self.evaluator.eval(self.problem, off, algorithm=self)

        # randomly assign an offspring to a nest to be replaced
        # I = perm()[:len(off)]
        I = off.get("index")

        # replace the solution in each nest where it has improved
        has_improved = ImprovementReplacement().do(self.problem, pop[I], off, return_indices=True)

        # choose some nests to be abandon no matter if they have improved or not
        abandon = np.random.random(len(I)) < self.pa

        # never abandon the currently best solution
        abandon[I == best] = False

        # either because they have improved or they have been abandoned they are replaced
        replace = has_improved | abandon

        # replace the individuals in the population
        self.pop[I[replace]] = off[replace]
        self.off = off


parse_doc_string(CuckooSearch.__init__)
