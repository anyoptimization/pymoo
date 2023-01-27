import math
from copy import deepcopy

import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.individual import Individual
from pymoo.core.infill import InfillCriterion
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.variable import Choice, Real, Integer, Binary, BoundedVariable
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.bitflip import BFM
from pymoo.operators.mutation.pm import PM
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.util.display.single import SingleObjectiveOutput


class MixedVariableMating(InfillCriterion):

    def __init__(self,
                 selection=RandomSelection(),
                 crossover=None,
                 mutation=None,
                 repair=None,
                 eliminate_duplicates=True,
                 n_max_iterations=100,
                 **kwargs):

        super().__init__(repair, eliminate_duplicates, n_max_iterations, **kwargs)

        if crossover is None:
            crossover = {
                Binary: UX(),
                Real: SBX(),
                Integer: SBX(vtype=float, repair=RoundingRepair()),
                Choice: UX(),
            }

        if mutation is None:
            mutation = {
                Binary: BFM(),
                Real: PM(),
                Integer: PM(vtype=float, repair=RoundingRepair()),
                Choice: ChoiceRandomMutation(),
            }

        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

    def _do(self, problem, pop, n_offsprings, parents=False, **kwargs):

        # So far we assume all crossover need the same amount of parents and create the same number of offsprings
        XOVER_N_PARENTS = 2
        XOVER_N_OFFSPRINGS = 2

        # the variables with the concrete information
        vars = problem.vars

        # group all the variables by their types
        vars_by_type = {}
        for k, v in vars.items():
            clazz = type(v)

            if clazz not in vars_by_type:
                vars_by_type[clazz] = []
            vars_by_type[clazz].append(k)

        # # all different recombinations (the choices need to be split because of data types)
        recomb = []
        for clazz, list_of_vars in vars_by_type.items():
            if clazz == Choice:
                for e in list_of_vars:
                    recomb.append((clazz, [e]))
            else:
                recomb.append((clazz, list_of_vars))

        # create an empty population that will be set in each iteration
        off = Population.new(X=[{} for _ in range(n_offsprings)])

        if not parents:
            n_select = math.ceil(n_offsprings / XOVER_N_OFFSPRINGS)
            pop = self.selection(problem, pop, n_select, XOVER_N_PARENTS, **kwargs)

        for clazz, list_of_vars in recomb:

            crossover = self.crossover[clazz]
            assert crossover.n_parents == XOVER_N_PARENTS and crossover.n_offsprings == XOVER_N_OFFSPRINGS

            _parents = [[Individual(X=np.array([parent.X[var] for var in list_of_vars])) for parent in parents] for
                        parents in pop]

            _vars = {e: vars[e] for e in list_of_vars}
            _xl = np.array([vars[e].lb if hasattr(vars[e], "lb") else None for e in list_of_vars])
            _xu = np.array([vars[e].ub if hasattr(vars[e], "ub") else None for e in list_of_vars])
            _problem = Problem(vars=_vars, xl=_xl, xu=_xu)

            _off = crossover(_problem, _parents, **kwargs)

            mutation = self.mutation[clazz]
            _off = mutation(_problem, _off, **kwargs)

            for k in range(n_offsprings):
                for i, name in enumerate(list_of_vars):
                    off[k].X[name] = _off[k].X[i]

        return off


class MixedVariableSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        V = {name: var.sample(n_samples) for name, var in problem.vars.items()}

        X = []
        for k in range(n_samples):
            X.append({name: V[name][k] for name in problem.vars.keys()})

        return X


class MixedVariableDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        a, b = a.X, b.X
        for k, v in a.items():
            if k not in b or b[k] != v:
                return False
        return True


def groups_of_vars(vars):
    ret = {}
    for name, var in vars.items():
        if var.__class__ not in ret:
            ret[var.__class__] = []

        ret[var.__class__].append((name, var))

    return ret


class MixedVariableGA(GeneticAlgorithm):

    def __init__(self,
                 pop_size=50,
                 n_offsprings=None,
                 output=SingleObjectiveOutput(),
                 sampling=MixedVariableSampling(),
                 mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                 eliminate_duplicates=MixedVariableDuplicateElimination(),
                 survival=FitnessSurvival(),
                 **kwargs):
        super().__init__(pop_size=pop_size, n_offsprings=n_offsprings, sampling=sampling, mating=mating,
                         eliminate_duplicates=eliminate_duplicates, output=output, survival=survival, **kwargs)
