from pymoo.core.problem import MetaProblem


class MultiToSingleObjective(MetaProblem):

    def __init__(self, problem, decomposition, kwargs=None):
        super().__init__(problem)
        
        self.decomposition = decomposition
        self.kwargs = kwargs if kwargs is not None else dict()
        self.n_obj = 1

    def do(self, X, return_values_of, *args, **kwargs):
        out = self.__wrapped__.do(X, return_values_of, *args, **kwargs)
        F = out["F"]
        out["__F__"] = F
        out["F"] = self.decomposition.do(F, **self.kwargs)[:, None]
        return out




