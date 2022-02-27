"""

Differential Evolution (DE)

-------------------------------- Description -------------------------------



-------------------------------- References --------------------------------

[1] J. Blank and K. Deb, pymoo: Multi-Objective Optimization in Python, in IEEE Access,
vol. 8, pp. 89497-89509, 2020, DOI: 10.1109/ACCESS.2020.2990567

-------------------------------- License -----------------------------------


----------------------------------------------------------------------------
"""

import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.infill import InfillCriterion
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling
from pymoo.core.parameters import get_params, flatten
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.replacement import ImprovementReplacement
from pymoo.core.variable import Choice, get, Binary
from pymoo.core.variable import Real
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.binx import mut_binomial
from pymoo.operators.crossover.expx import mut_exp
from pymoo.operators.mutation.pm import PM
from pymoo.operators.param_control import EvolutionaryParameterControl, AgeBasedTournamentSelection, NoParameterControl
from pymoo.operators.repair.bounds_repair import repair_random_init
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.rnd import fast_fill_random
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.misc import where_is_what
from pymoo.util.termination.default import SingleObjectiveDefaultTermination


# =========================================================================================================
# Crossover
# =========================================================================================================

def de_differential(X, F, jitter, alpha=0.001):
    n_parents, n_matings, n_var = X.shape
    assert n_parents % 2 == 1, "For the differential an odd number of values need to be provided"

    # the differentials from each pair
    delta = np.zeros((n_matings, n_var))

    # for each difference of the differences
    for i in range(1, n_parents, 2):
        # create the weight vectors with jitter to give some variation
        _F = F[:, None].repeat(n_var, axis=1)
        _F[jitter] *= (1 + alpha * (np.random.random((jitter.sum(), n_var)) - 0.5))

        # add the difference to the vector
        delta += _F * (X[i] - X[i + 1])

    # now add the differentials to the first parent
    Xp = X[0] + delta

    return Xp


# =========================================================================================================
# Different Variants of Differential Evolution
# =========================================================================================================


class Variant(InfillCriterion):

    def __init__(self,
                 selection="best",
                 n_diffs=1,
                 F=0.5,
                 crossover="bin",
                 CR=0.75,
                 jitter=False,
                 prob_mut=0.1,
                 **kwargs):

        super().__init__(**kwargs)
        self.selection = Choice(selection, options=["best"], all=["rand", "best", "target-to-best"])
        self.n_diffs = Choice(n_diffs, options=[1], all=[1, 2])
        self.F = Real(F, bounds=(0.4, 0.6), strict=(0.0, None))
        self.crossover = Choice(crossover, ["bin"], all=["bin", "exp", "hypercube", "line"])
        self.CR = Real(CR, bounds=(0.3, 0.7), strict=(0.0, 1.0))
        self.jitter = Choice(jitter, options=[False], all=[True, False])

        self.mutation = PM(at_least_once=True)
        self.mutation.eta = 20
        # self.mutation.prob = prob_mut
        self.mutation.prob = Real(prob_mut, bounds=(0.05, 0.35))
        # self.mutation.prob_var = Real(None, bounds=(0.0, 0.5))

    def do(self, problem, pop, n_offsprings, algorithm=None, **kwargs):

        # find the different groups of selection schemes and order them by category
        sel, n_diffs = get(self.selection, self.n_diffs, size=n_offsprings)
        H = where_is_what(zip(sel, n_diffs))

        # get the parameters used for reproduction during the crossover
        F, CR, jitter = get(self.F, self.CR, self.jitter, size=n_offsprings)

        # the `target` vectors which will be recombined
        X = pop.get("X")

        # the `donor` vector which will be obtained through the differential equation
        donor = np.empty_like(X)

        # for each type defined by the type and number of differentials
        for (sel_type, n_diffs), targets in H.items():

            # the number of offsprings created in this run
            n_matings, n_parents = len(targets), 1 + 2 * n_diffs

            # create the parents array
            P = np.full([n_matings, n_parents], -1)

            itself = np.array(targets)[:, None]

            best = lambda: np.random.choice(np.where(pop.get("rank") == 0)[0], replace=True, size=n_matings)

            if sel_type == "rand":
                fast_fill_random(P, len(pop), columns=range(n_parents), Xp=itself)
            elif sel_type == "best":
                P[:, 0] = best()
                fast_fill_random(P, len(pop), columns=range(1, n_parents), Xp=itself)
            elif sel_type == "target-to-best":
                P[:, 0] = targets
                P[:, 1] = best()
                fast_fill_random(P, len(pop), columns=range(2, n_parents), Xp=itself)
            else:
                raise Exception("Unknown selection method.")

            # get the values of the parents in the design space
            XX = np.swapaxes(X[P], 0, 1)

            # do the differential crossover to create the donor vector
            Xp = de_differential(XX, F[targets], jitter[targets])

            # make sure everything stays in bounds
            if problem.has_bounds():
                Xp = repair_random_init(Xp, XX[0], *problem.bounds())

            # set the donors (the one we have created in this step)
            donor[targets] = Xp

        # the `trial` created by by recombining target and donor
        trial = np.empty_like(X)

        crossover = get(self.crossover, size=n_offsprings)
        for name, K in where_is_what(crossover).items():

            _target = X[K]
            _donor = donor[K]
            _CR = CR[K]

            if name == "bin":
                M = mut_binomial(len(K), problem.n_var, _CR, at_least_once=True)
                _trial = np.copy(_target)
                _trial[M] = _donor[M]
            elif name == "exp":
                M = mut_exp(n_offsprings, problem.n_var, _CR, at_least_once=True)
                _trial = np.copy(_target)
                _trial[M] = _donor[M]
            elif name == "line":
                w = np.random.random((len(K), 1)) * _CR[:, None]
                _trial = _target + w * (_donor - _target)
            elif name == "hypercube":
                w = np.random.random((len(K), _target.shape[1])) * _CR[:, None]
                _trial = _target + w * (_donor - _target)
            else:
                raise Exception(f"Unknown crossover variant: {name}")

            trial[K] = _trial

        # create the population
        off = Population.new(X=trial)

        # do the mutation which helps to add some more diversity
        off = self.mutation.do(problem, off)

        # repair the individuals if necessary - disabled if repair is NoRepair
        off = self.repair.do(problem, off, **kwargs)

        return off


# =========================================================================================================
# Implementation
# =========================================================================================================


class DE(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 n_offsprings=None,
                 sampling=FloatRandomSampling(),
                 variant=None,
                 display=SingleObjectiveDisplay(),
                 control=None,
                 **kwargs
                 ):

        if variant is None:
            if "control" not in kwargs:
                kwargs["control"] = EvolutionaryParameterControl
            variant = Variant(**kwargs)

        elif isinstance(variant, str):
            try:
                _, selection, n_diffs, crossover = variant.split("/")
                variant = Variant(selection=selection, n_diffs=n_diffs, crossover=crossover, **kwargs)
            except:
                raise Exception("Please provide a valid variant: DE/<selection>/<n_diffs>/<crossover>")

        super().__init__(pop_size=pop_size,
                         n_offsprings=n_offsprings,
                         sampling=sampling,
                         mating=variant,
                         survival=None,
                         display=display,
                         eliminate_duplicates=False,
                         **kwargs)

        self.default_termination = SingleObjectiveDefaultTermination()

        self.control = control
        self.prev_params = None
        self.params = None

    def _initialize_advance(self, infills=None, **kwargs):
        FitnessSurvival().do(self.problem, self.pop, return_indices=True)

    def _infill(self):

        if self.control is not NoParameterControl:

            omega = flatten(get_params(self.mating))

            problem = Problem(vars=omega)

            if self.prev_params is None:
                self.params = MixedVariableSampling().do(problem, self.pop_size)
                self.prev_params = Population.create(*self.params)
            else:
                selection = AgeBasedTournamentSelection()
                mating = MixedVariableMating(selection=selection, eliminate_duplicates=NoDuplicateElimination())
                self.params = mating.do(problem, self.prev_params, self.n_offsprings)

            self.params.set("n_gen", self.n_gen)

            P = self.params.get("X")
            for name, param in omega.items():
                param.set(np.array([d[name] for d in P]))

        infills = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

        return infills

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."
        prev_pop = Population.create(*self.pop)

        # replace the individuals with the corresponding parents from the mating
        ImprovementReplacement().do(self.problem, self.pop, infills, inplace=True)

        # update the information regarding the current population
        FitnessSurvival().do(self.problem, self.pop)

        if self.control is not NoParameterControl:

            has_not_improved = self.pop == prev_pop
            self.params[has_not_improved] = self.prev_params[has_not_improved]
            self.prev_params = self.params

    def _set_optimum(self, **kwargs):
        k = self.pop.get("rank") == 0
        self.opt = self.pop[k]


parse_doc_string(DE.__init__)
