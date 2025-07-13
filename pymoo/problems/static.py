from pymoo.core.problem import MetaProblem


class StaticProblem(MetaProblem):

    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        self.kwargs = kwargs

    def _evaluate(self, _, out, *args, **kwargs):
        for K, V in self.kwargs.items():
            out[K] = V

