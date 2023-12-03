from typing import Generator

from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluation
from pymoo.core.fitness import get_fittest
from pymoo.core.problem import Problem
from pymoo.core.solution import SolutionSet, Solution
from pymoo.operators.sampling import Sampling


class LocalSearch(Algorithm):

    def __init__(self, n_init_sols=20, **kwargs):
        super().__init__(**kwargs)
        self.n_init_sols = n_init_sols
        self.init_sol = None

    def setup(self,
              problem: Problem,
              init_sol: Solution = None,
              **kwargs) -> Algorithm:
        super().setup(problem, **kwargs)
        self.init_sol = init_sol
        return self

    def initialize(self) -> Generator[Evaluation, None, SolutionSet]:
        init_sol = self.init_sol

        if init_sol is None:

            if init_sol is None:
                random = Sampling().sample(self.problem, size=self.n_init_sols)
                init_sols = yield from self.evaluator.send(random)

                init_sol = get_fittest(init_sols)

        else:
            init_sol = yield from self.evaluator.send(self.init_sol)

        self.init_sol = init_sol
        return SolutionSet([init_sol])


