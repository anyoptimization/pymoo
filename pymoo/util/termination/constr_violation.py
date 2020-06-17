from pymoo.model.termination import Termination
from pymoo.util.misc import to_numpy
from pymoo.util.termination.sliding_window_termination import SlidingWindowTermination


class ConstraintViolationToleranceTermination(SlidingWindowTermination):

    def __init__(self,
                 n_last=20,
                 tol=1e-6,
                 nth_gen=1,
                 n_max_gen=None,
                 n_max_evals=None,
                 **kwargs):

        super().__init__(metric_window_size=n_last,
                         data_window_size=2,
                         min_data_for_metric=2,
                         nth_gen=nth_gen,
                         n_max_gen=n_max_gen,
                         n_max_evals=n_max_evals,
                         **kwargs)
        self.tol = tol

    def _store(self, algorithm):
        return algorithm.opt.get("CV").max()

    def _metric(self, data):
        last, current = data[-2], data[-1]
        return {"cv": current,
                "delta_cv": abs(last - current)
                }

    def _decide(self, metrics):
        cv = to_numpy([e["cv"] for e in metrics])
        delta_cv = to_numpy([e["delta_cv"] for e in metrics])
        n_feasible = (cv <= 0).sum()

        # if the whole window had only feasible solutions
        if n_feasible == len(metrics):
            return False
        # transition period - some were feasible some were not
        elif 0 < n_feasible < len(metrics):
            return True
        # all solutions are infeasible
        else:
            return delta_cv.max() > self.tol


class FeasibleSolutionFoundTermination(Termination):

    def _do_continue(self, algorithm):
        return algorithm.opt.get("CV").min() != 0
