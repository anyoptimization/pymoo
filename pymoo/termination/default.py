from pymoo.core.termination import Termination
from pymoo.termination.cv import ConstraintViolationTermination
from pymoo.termination.ftol import SingleObjectiveSpaceTermination, MultiObjectiveSpaceTermination
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.robust import RobustTermination
from pymoo.termination.xtol import DesignSpaceTermination


class DefaultTermination(Termination):

    def __init__(self, x, cv, f, n_max_gen=1000, n_max_evals=100000) -> None:
        super().__init__()
        self.x = x
        self.cv = cv
        self.f = f

        self.max_gen = MaximumGenerationTermination(n_max_gen)
        self.max_evals = MaximumFunctionCallTermination(n_max_evals)

        self.criteria = [self.x, self.cv, self.f, self.max_gen, self.max_evals]

    def _update(self, algorithm):
        p = [criterion.update(algorithm) for criterion in self.criteria]
        return max(p)


class DefaultSingleObjectiveTermination(DefaultTermination):

    def __init__(self, xtol=1e-8, cvtol=1e-8, ftol=1e-6, period=30, **kwargs) -> None:
        x = RobustTermination(DesignSpaceTermination(xtol), period=period)
        cv = RobustTermination(ConstraintViolationTermination(cvtol, terminate_when_feasible=False), period=period)
        f = RobustTermination(SingleObjectiveSpaceTermination(ftol, only_feas=True), period=period)
        super().__init__(x, cv, f, **kwargs)


class DefaultMultiObjectiveTermination(DefaultTermination):

    def __init__(self, xtol=0.0005, cvtol=1e-8, ftol=0.005, n_skip=5, period=50, **kwargs) -> None:
        x = RobustTermination(DesignSpaceTermination(tol=xtol, n_skip=n_skip), period)
        cv = RobustTermination(ConstraintViolationTermination(cvtol, terminate_when_feasible=False, n_skip=n_skip), period)
        f = RobustTermination(MultiObjectiveSpaceTermination(ftol, only_feas=True, n_skip=n_skip), period)
        super().__init__(x, cv, f, **kwargs)


