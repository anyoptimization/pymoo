from pymoo.util.termination.constr_violation import ConstraintViolationToleranceTermination
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination, \
    SingleObjectiveSpaceToleranceTermination
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination
from pymoo.util.termination.max_gen import MaximumGenerationTermination
from pymoo.util.termination.tolerance import ToleranceBasedTermination
from pymoo.util.termination.x_tol import DesignSpaceToleranceTermination


class DefaultTermination(ToleranceBasedTermination):

    def __init__(self,
                 x_tol,
                 cv_tol,
                 f_tol,
                 nth_gen=5,
                 n_last=20,
                 n_max_gen=1000,
                 n_max_evals=100000,
                 **kwargs) -> None:
        """

        Parameters
        ----------
        cv_tol
        x_tol
        f_tol

        n_max_gen : int
            A limit on maximum number of generations to avoid it never terminates
            (set to `None` or infinity to disable)

        n_max_evals : int
            A limit on maximum number of function evaluations to avoid it never terminates
            (set to `None` or infinity to disable)

        nth_gen
        n_last
        kwargs
        """

        super().__init__(n_last=n_last,
                         nth_gen=nth_gen,
                         **kwargs)

        self.n_max_gen = MaximumGenerationTermination(n_max_gen)
        self.n_max_evals = MaximumFunctionCallTermination(n_max_evals)

        self.x_tol = DesignSpaceToleranceTermination(tol=x_tol, n_last=n_last, nth_gen=nth_gen)

        self.cv = ConstraintViolationToleranceTermination(tol=cv_tol, n_last=n_last, nth_gen=nth_gen)
        self.f_tol = f_tol

    def _store(self, algorithm):
        return {
            "n_max_gen": self.n_max_gen.do_continue(algorithm),
            "n_max_evals": self.n_max_evals.do_continue(algorithm),
            "x_tol": self.x_tol.do_continue(algorithm),
            "cv": self.cv.do_continue(algorithm),
            "f_tol": self.f_tol.do_continue(algorithm)
        }

    def _decide(self):
        metric = self.history[-1]
        return metric["n_max_gen"] and metric["n_max_evals"] and metric["x_tol"] and (metric["cv"] or metric["f_tol"])


class SingleObjectiveDefaultTermination(DefaultTermination):

    def __init__(self,
                 x_tol=1e-6,
                 cv_tol=1e-6,
                 f_tol=1e-6,
                 nth_gen=5,
                 n_last=20,
                 **kwargs) -> None:

        super().__init__(cv_tol,
                         x_tol,
                         SingleObjectiveSpaceToleranceTermination(tol=f_tol, n_last=n_last, nth_gen=nth_gen),
                         nth_gen,
                         n_last,
                         **kwargs)


class MultiObjectiveDefaultTermination(DefaultTermination):
    def __init__(self,
                 x_tol=1e-6,
                 cv_tol=1e-6,
                 f_tol=0.0025,
                 nth_gen=5,
                 n_last=30,
                 **kwargs) -> None:

        super().__init__(cv_tol,
                         x_tol,
                         MultiObjectiveSpaceToleranceTermination(tol=f_tol, n_last=n_last, nth_gen=nth_gen),
                         nth_gen,
                         n_last,
                         **kwargs)

