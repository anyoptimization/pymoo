import numpy as np

from pymoo.model.termination import Termination
from pymoo.util.termination.tolerance import ToleranceBasedTermination


class ConstraintViolationToleranceTermination(ToleranceBasedTermination):

    def __init__(self,
                 tol=0.0001,
                 **kwargs) -> None:

        super().__init__(n_hist_at_least=2, n_hist=kwargs["n_last"], **kwargs)
        self.tol = tol

    def _store(self, algorithm):
        return algorithm.opt.get("CV")[:, 0].min(axis=0)

    def _calc_metric(self):
        CV = np.array([e for e in self.history])

        # calculate the improvement regarding the CV in each transition
        delta_CV = np.array([CV[k] - CV[k + 1] for k in range(len(CV) - 1)])

        return {
            "some_feasible": CV.min() == 0,
            "all_feasible": CV.max() == 0,
            "delta_cv": delta_CV.max()
        }

    def _decide(self):
        metric = self.metrics[-1]

        # if all are feasible then no need to continue
        if metric["all_feasible"]:
            return False

        # if not all feasible, but some are feasible in the window - just a matter of time until all are feasible
        elif metric["some_feasible"]:
            return True

        # otherwise look at the improvement from the last generations
        else:
            return metric["delta_cv"] > self.tol


class FeasibleSolutionFoundTermination(Termination):

    def _do_continue(self, algorithm):
        return algorithm.opt.get("CV").min() != 0
