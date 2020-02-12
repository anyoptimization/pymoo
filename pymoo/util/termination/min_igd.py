from pymoo.model.termination import Termination
from pymoo.performance_indicator.igd import IGD


class IGDTermination(Termination):

    def __init__(self, min_igd, pf) -> None:
        super().__init__()
        if pf is None:
            raise Exception("You can only use IGD termination criteria if the pareto front is known!")

        self.perf = IGD(pf)
        self.min_igd = min_igd

    def _do_continue(self, algorithm):
        F = algorithm.opt.get("F")
        return self.perf.calc(F) > self.min_igd

