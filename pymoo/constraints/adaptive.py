from copy import deepcopy

import numpy as np

from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
from pymoo.core.meta import Meta
from pymoo.core.population import Population
from pymoo.core.problem import Problem


class AttachConfigEvaluator(Meta, Evaluator):

    def __init__(self, wrapped, config):
        super().__init__(wrapped)
        self.config = config

    def eval(self, problem: Problem, pop: Population, **kwargs):
        pop = super().eval(problem, pop, **kwargs)
        pop.apply(lambda ind: ind.set("config", self.config))


def copy_to_dict(src, dest):
    dest.clear()
    dest.update(**src)


class AdaptiveConstraintHandling(Meta, Algorithm):

    def __init__(self, algorithm):
        super().__init__(algorithm)

        self.config = Individual.default_config()
        self.config["cache"] = False

        self.default_config = deepcopy(self.config)
        self.adapted_config = deepcopy(self.config)

    def _setup(self, _, **kwargs):
        self.evaluator = AttachConfigEvaluator(self.evaluator, self.config)

    def _adapt(self, config, infills=None, **kwargs):
        pass

    def _initialize_advance(self, infills=None, **kwargs):
        copy_to_dict(self.adapted_config, self.config)
        super()._initialize_advance(infills=infills, **kwargs)
        copy_to_dict(self.default_config, self.config)

    def _advance(self, infills=None, **kwargs):
        copy_to_dict(self.adapted_config, self.config)
        super()._advance(infills=infills, **kwargs)
        copy_to_dict(self.default_config, self.config)

        self._adapt(self.adapted_config, infills=infills, **kwargs)

    def _infill(self):
        copy_to_dict(self.adapted_config, self.config)
        pop = super()._infill()
        copy_to_dict(self.default_config, self.config)
        return pop
