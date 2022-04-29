from pymoo.core.termination import Termination
from pymoo.util.termination.cv import ConstraintViolationTermination
from pymoo.util.termination.ftol import SingleObjectiveSpaceTermination, MultiObjectiveSpaceTermination
from pymoo.util.termination.robust import RobustTermination
from pymoo.util.termination.xtol import DesignSpaceTermination


class DefaultTermination(Termination):

    def __init__(self, x, cv, f) -> None:
        super().__init__()
        self.x = x
        self.cv = cv
        self.f = f

    def _update(self, algorithm):
        cv = self.cv.update(algorithm)
        x = self.x.update(algorithm)
        f = self.f.update(algorithm)
        return min(cv, max(x, f))


class DefaultSingleObjectiveTermination(DefaultTermination):

    def __init__(self) -> None:
        x = RobustTermination(DesignSpaceTermination(1e-8), 30)
        cv = RobustTermination(ConstraintViolationTermination(1e-8), 50)
        f = RobustTermination(SingleObjectiveSpaceTermination(1e-6), 30)
        super().__init__(x, cv, f)


class DefaultMultiObjectiveTermination(DefaultTermination):

    def __init__(self, n_skip=5) -> None:
        x = RobustTermination(DesignSpaceTermination(1e-8, n_skip=n_skip), 30)
        cv = RobustTermination(ConstraintViolationTermination(1e-8, n_skip=n_skip), 50)
        f = RobustTermination(MultiObjectiveSpaceTermination(0.0025, n_skip=n_skip), 50)
        super().__init__(x, cv, f)


