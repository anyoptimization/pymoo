
from pymoo.core.termination import Termination


class TerminationCollection(Termination):

    def __init__(self, *args) -> None:
        super().__init__()
        self.terminations = args

    def _update(self, algorithm):
        return max([termination.update(algorithm) for termination in self.terminations])
