from pymoo.model.termination import Termination


class MaximumFunctionCallTermination(Termination):

    def __init__(self, n_max_evals) -> None:
        super().__init__()
        self.n_max_evals = n_max_evals

        if self.n_max_evals is None:
            self.n_max_evals = float("inf")

    def _do_continue(self, algorithm):
        return algorithm.evaluator.n_eval < self.n_max_evals

