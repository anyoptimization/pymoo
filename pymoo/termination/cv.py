from pymoo.core.termination import Termination
from pymoo.termination.delta import DeltaToleranceTermination


class ConstraintViolationTermination(DeltaToleranceTermination):

    def __init__(self, tol=1e-6, **kwargs):
        super().__init__(tol, **kwargs)

    def _update(self, algorithm):
        if algorithm.problem.has_constraints():
            return super()._update(algorithm)
        else:
            return 1.0

    def _delta(self, prev, current):
        return max(0, prev - current)

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
