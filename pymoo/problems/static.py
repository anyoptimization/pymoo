from pymoo.problems.meta import MetaProblem


class StaticProblem(MetaProblem):

    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        self.kwargs = kwargs

    def _evaluate(self, x, out, *args, **kwargs):
        for K, V in self.kwargs.items():
            out[K] = V

