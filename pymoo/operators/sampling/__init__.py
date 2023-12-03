from pymoo.core.problem import Problem

from pymoo.core.solution import Solution, SolutionSet


class Sampling:

    def __init__(self):
        super().__init__()

    def sample(self,
               problem: Problem,
               size: int):
        vtype = problem.vtype
        return SolutionSet([Solution(var=vtype.random()) for _ in range(size)])

