from pymoo.core.termination import Termination
from pymoo.termination.delta import DeltaToleranceTermination


class ConstraintViolationTermination(DeltaToleranceTermination):

    def __init__(self, tol=1e-6, terminate_when_feasible=True, **kwargs):
        super().__init__(tol, **kwargs)
        self.terminate_when_feasible = terminate_when_feasible

    def _update(self, algorithm):
        if algorithm.problem.has_constraints():
            feasible_found = any(algorithm.opt.get("feas"))

            if feasible_found:
                if self.terminate_when_feasible:
                    return 1.0
                else:
                    return 0.0

            else:
                return super()._update(algorithm)
        else:
            return 0.0

    def _delta(self, prev, current):
        return max(0.0, prev - current)

    def _data(self, algorithm):
        return algorithm.opt.get("CV").min()


class UntilFeasibleTermination(Termination):

    def __init__(self) -> None:
        super().__init__()
        self.initial_cv = None

    def _update(self, algorithm):
        cv = algorithm.opt.get("CV").min()

        if self.initial_cv is None:
            if cv <= 0:
                self.initial_cv = 1e-32
            else:
                self.initial_cv = cv

        return 1 - cv / self.initial_cv
