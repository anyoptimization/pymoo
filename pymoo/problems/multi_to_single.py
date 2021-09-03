from pymoo.problems.meta import MetaProblem


class MultiToSingleObjective(MetaProblem):

    def __init__(self, problem, decomposition, kwargs=None):
        super().__init__(problem)
        self.decomposition = decomposition
        self.kwargs = kwargs if not None else dict()

        self.n_obj = 1

    def do(self, x, out, *args, **kwargs):
        super().do(x, out, *args, **kwargs)

        F = out["F"]
        out["__F__"] = F
        out["F"] = self.decomposition.do(F, **self.kwargs)[:, None]




