from pymoo.model.termination import Termination


class MaximumGenerationTermination(Termination):

    def __init__(self, n_max_gen) -> None:
        super().__init__()
        self.n_max_gen = n_max_gen

        if self.n_max_gen is None:
            self.n_max_gen = float("inf")

    def _do_continue(self, algorithm):
        return algorithm.n_gen < self.n_max_gen

