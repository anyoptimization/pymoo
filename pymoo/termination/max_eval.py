from pymoo.core.termination import Termination


class MaximumFunctionCallTermination(Termination):

    def __init__(self, n_max_evals=float("inf")) -> None:
        super().__init__()
        self.n_max_evals = n_max_evals

    def _update(self, algorithm):
        if self.n_max_evals is None:
            return 0.0
        else:
            return algorithm.evaluator.n_eval / self.n_max_evals
