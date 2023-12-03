from collections.abc import Generator

import numpy as np

from pymoo.core.parallelize import Serial, Parallelize
from pymoo.core.problem import Problem
from pymoo.core.solution import SolutionSet, Solution
from pymoo.core.variable import Variable


class Evaluation:

    def __init__(self,
                 sols: SolutionSet,
                 problem: Problem = None,
                 rtype=SolutionSet,
                 parallelize: Parallelize = Serial()):
        super().__init__()
        self.sols = sols
        self.problem = problem
        self.rtype = rtype
        self.parallelize = parallelize

    def solution(self) -> Solution:
        if len(self.sols) > 0:
            return self.sols.item(0)

    def solutions(self) -> SolutionSet:
        return self.sols

    def get(self):
        if self.rtype == SolutionSet:
            return self.solutions()
        elif self.rtype == Solution:
            return self.solution()
        else:
            raise Exception("Unknown return type.")

    def run(self):
        problem = self.problem
        assert problem is not None, "To run the evaluation a problem needs to be given."

        if problem.parallel:
            assert False, "Not supported yet."

        else:

            def f(sol):
                out = problem.evaluate(sol.var)
                for k, v in out.items():
                    setattr(sol, k, v)

            self.parallelize(f, self.solutions())


class Evaluator:

    def __init__(self,
                 problem: Problem,
                 parallelize: Parallelize = Serial()
                 ):
        super().__init__()
        self.problem = problem
        self.parallelize = parallelize
        self.fevals = 0

    def create(self, obj: np.ndarray | Solution | SolutionSet):

        problem = self.problem
        vtype = problem.vtype
        rtype = SolutionSet

        if isinstance(obj, SolutionSet):
            sols = obj

        elif isinstance(obj, Solution):
            sols = SolutionSet([obj])
            rtype = Solution

        elif isinstance(obj, Variable):
            sols = SolutionSet([obj.solution()])
            rtype = Solution

        elif isinstance(obj, np.ndarray):

            if obj.ndim == 1:
                sols = SolutionSet([vtype.new(obj).solution()])
                rtype = Solution
            elif obj.ndim == 2:
                sols = SolutionSet([vtype.new(row).solution() for row in obj])
            else:
                raise Exception("Either provide a 1 or 2 dimensional array.")

        else:
            sols = SolutionSet([vtype.new(obj).solution()])
            rtype = Solution

        evaluation = Evaluation(sols, problem=self.problem, rtype=rtype, parallelize=self.parallelize)

        return evaluation

    def send(self, obj: np.ndarray | Solution | SolutionSet) -> Generator[Evaluation, None, Solution | SolutionSet]:
        evaluation = self.create(obj)

        self.fevals += len(evaluation.solutions())

        yield evaluation
        return evaluation.get()
