from pymoo.model.termination import Termination


class NoTermination(Termination):

    def _do_continue(self, algorithm, **kwargs):
        return True
