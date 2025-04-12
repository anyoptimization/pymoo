from pymoo.core.meta import Meta
from pymoo.core.problem import Problem


class StaticProblem(Meta, Problem):

    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        self.kwargs = kwargs

    def _evaluate(self, _, out, *args, **kwargs):
        for K, V in self.kwargs.items():
            out[K] = V

