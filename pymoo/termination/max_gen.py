from pymoo.core.termination import Termination


class MaximumGenerationTermination(Termination):

    def __init__(self, n_max_gen=float("inf")) -> None:
        super().__init__()
        self.n_max_gen = n_max_gen

    def _update(self, algorithm):
        return algorithm.n_gen / self.n_max_gen

