from pymoo.indicators.igd import IGD


class Termination:

    def __init__(self) -> None:
        super().__init__()
        self.flag = True

    def do_continue(self, D):
        return self.flag and self._do_continue(D)

    def has_finished(self, D):
        return not self.do_continue(D)


class MaximumFunctionCallTermination(Termination):

    def __init__(self, n_max_evals) -> None:
        super().__init__()
        self.n_max_evals = n_max_evals

    def _do_continue(self, algorithm):
        return algorithm.evaluator.n_eval < self.n_max_evals


class MaximumGenerationTermination(Termination):

    def __init__(self, n_max_gen) -> None:
        super().__init__()
        self.n_max_gen = n_max_gen

    def _do_continue(self, algorithm):
        return algorithm.n_gen < self.n_max_gen


class IGDTermination(Termination):

    def __init__(self, min_igd, pf) -> None:
        super().__init__()
        if pf is None:
            raise Exception("You can only use IGD termination criteria if the pareto front is known!")

        self.obj = IGD(pf)
        self.igd = min_igd

    def _do_continue(self, algorithm):
        F = algorithm.pop.get("F")
        return self.obj.calc(F) > self.igd

