import math

import numpy as np

from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.infill import InfillCriterion
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.variable import Choice, Real, Integer, Binary
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX
from pymoo.operators.integer_from_float_operator import IntegerFromFloatCrossover, IntegerFromFloatMutation
from pymoo.operators.mutation.bitflip import BFM
from pymoo.operators.mutation.pm import PM
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.operators.selection.rnd import RandomSelection


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
                Integer: IntegerFromFloatCrossover(SBX),
                Choice: UX(),
            }

        if mutation is None:
            mutation = {
                Binary: BFM(),
                Real: PM(),
                Integer: IntegerFromFloatMutation(PM),
                Choice: ChoiceRandomMutation(),
            }

        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):
        X = pop.get("X")

        vars = problem.vars

        vars_by_type = {}
        for k, v in vars.items():
            clazz = type(v)
            if clazz not in vars_by_type:
                vars_by_type[clazz] = []
            vars_by_type[clazz].append(k)

        off = Population.new(X=[{} for _ in range(n_offsprings)])

        cross_n_offsprings = 2
        cross_n_parents = 2

        if parents is None:
            n_select = math.ceil(n_offsprings / cross_n_offsprings)
            parents = self.selection._do(problem, pop, n_select, cross_n_parents, **kwargs)

        for clazz, list_of_vars in vars_by_type.items():
            _X = np.array([[x[var] for var in list_of_vars] for x in X], dtype=clazz.type)
            _pop = Population.new(X=_X)

            crossover = self.crossover[clazz]
            assert crossover.n_parents == 2 and crossover.n_offsprings == 2

            _vars = [vars[e] for e in list_of_vars]
            _xl, _xu = None, None

            if clazz in [Real, Integer]:
                _xl, _xu = np.array([v.bounds for v in _vars]).T

            _problem = Problem(vars=_vars, xl=_xl, xu=_xu)

            _off = crossover.do(_problem, _pop, parents, **kwargs)

            mutation = self.mutation[clazz]
            _off = mutation.do(_problem, _off, **kwargs)

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
