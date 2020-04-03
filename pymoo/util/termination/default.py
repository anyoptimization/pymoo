from pymoo.util.termination.constr_violation import ConstraintViolationToleranceTermination
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
from pymoo.util.termination.f_tol_single import SingleObjectiveSpaceToleranceTermination
from pymoo.util.termination.sliding_window_termination import SlidingWindowTermination
from pymoo.util.termination.x_tol import DesignSpaceToleranceTermination


class DefaultTermination(SlidingWindowTermination):

    def __init__(self,
                 x_tol,
                 cv_tol,
                 f_tol,
                 n_max_gen=1000,
                 n_max_evals=100000,
                 **kwargs):
        super().__init__(metric_window_size=1,
                         data_window_size=1,
                         min_data_for_metric=1,
                         n_max_gen=n_max_gen,
                         n_max_evals=n_max_evals,
                         **kwargs)

        self.x_tol = x_tol
        self.cv_tol = cv_tol
        self.f_tol = f_tol

    def _store(self, algorithm):
        return algorithm

    def _metric(self, data):
        algorithm = data[-1]
        return {
            "x_tol": self.x_tol.do_continue(algorithm),
            "cv_tol": self.cv_tol.do_continue(algorithm),
            "f_tol": self.f_tol.do_continue(algorithm)
        }

    def _decide(self, metrics):
        decisions = metrics[-1]
        return decisions["x_tol"] and (decisions["cv_tol"] or decisions["f_tol"])


class SingleObjectiveDefaultTermination(DefaultTermination):

    def __init__(self,
                 x_tol=1e-8,
                 cv_tol=1e-6,
                 f_tol=1e-6,
                 nth_gen=5,
                 n_last=20,
                 **kwargs) -> None:
        super().__init__(DesignSpaceToleranceTermination(tol=x_tol, n_last=n_last),
                         ConstraintViolationToleranceTermination(tol=cv_tol, n_last=n_last),
                         SingleObjectiveSpaceToleranceTermination(tol=f_tol, n_last=n_last, nth_gen=nth_gen),
                         **kwargs)


class MultiObjectiveDefaultTermination(DefaultTermination):
    def __init__(self,
                 x_tol=1e-8,
                 cv_tol=1e-6,
                 f_tol=0.0025,
                 nth_gen=5,
                 n_last=30,
                 **kwargs) -> None:
        super().__init__(DesignSpaceToleranceTermination(tol=x_tol, n_last=n_last),
                         ConstraintViolationToleranceTermination(tol=cv_tol, n_last=n_last),
                         MultiObjectiveSpaceToleranceTermination(tol=f_tol, n_last=n_last, nth_gen=nth_gen),
                         **kwargs)
