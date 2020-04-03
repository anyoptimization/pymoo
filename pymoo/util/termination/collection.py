
from pymoo.model.termination import Termination


class TerminationCollection(Termination):

    def __init__(self, *args) -> None:
        super().__init__()
        self.terminations = args

    def _do_continue(self, algorithm):
        for term in self.terminations:
            if not term.do_continue(algorithm):
                return False
        return True
