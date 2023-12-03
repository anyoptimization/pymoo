from typing import Generator

from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluation
from pymoo.core.problem import Problem
from pymoo.core.solution import SolutionSet


class GoldenSectionSearch(Algorithm):

    def __init__(self, n_max_iter: int = 1000):
        super().__init__(n_max_iter)
        self.R = (5 ** 0.5 - 1) / 2

    def initialize(self) -> Generator[Evaluation, None, SolutionSet]:
        vtype = self.problem.vtype
        assert vtype.has_bounds(), "Either the problem has bounds or bounds or bounds are directly provided."

        a = yield from self.evaluator.send(vtype.low)
        b = yield from self.evaluator.send(vtype.high)

        # create the left and right in the interval itself
        c = yield from self.evaluator.send(b.X - self.R * (b.X - a.X))
        d = yield from self.evaluator.send(a.X + self.R * (b.X - a.X))

        # create a population with all four individuals
        sols = SolutionSet([a, c, d, b])

        return sols

    def advance(self) -> Generator[Evaluation, None, SolutionSet]:
        vtype = self.problem.vtype
        assert vtype.has_bounds(), "Either the problem has bounds or bounds or bounds are directly provided."

        # all the elements in the interval
        a, c, d, b = self.sols

        # if the left solution is better than the right
        if c.f < d.f:

            # make the right to be the new right bound and the left becomes the right
            a, b = a, d
            d = c

            # create a new left individual and evaluate
            c = yield from self.evaluator.send(b.X - self.R * (b.X - a.X))

        # if the right solution is better than the left
        else:

            # make the left to be the new left bound and the right becomes the left
            a, b = c, b
            c = d

            # create a new right individual and evaluate
            d = yield from self.evaluator.send(a.X + self.R * (b.X - a.X))

        # update the population with all the four individuals
        sols = SolutionSet([a, c, d, b])
        return sols
