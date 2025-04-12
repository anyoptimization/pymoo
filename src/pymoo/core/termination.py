from abc import abstractmethod


class Termination:

    def __init__(self) -> None:
        super().__init__()

        # the algorithm can be forced to terminate by setting this attribute to true
        self.force_termination = False

        # the value indicating how much perc has been made
        self.perc = 0.0

    def update(self, algorithm):
        """
        Provide the termination criterion a current status of the algorithm to update the perc.

        Parameters
        ----------
        algorithm : object
            The algorithm object which is used to determine whether a run has terminated.
        """

        if self.force_termination:
            progress = 1.0
        else:
            progress = self._update(algorithm)
            assert progress >= 0.0, "Invalid progress was set by the TerminationCriterion."

        self.perc = progress
        return self.perc

    def has_terminated(self):
        return self.perc >= 1.0

    def do_continue(self):
        return not self.has_terminated()

    def terminate(self):
        self.force_termination = True

    @abstractmethod
    def _update(self, algorithm):
        pass


class NoTermination(Termination):

    def _update(self, algorithm):
        return 0.0


class MultipleCriteria(Termination):

    def __init__(self, *args) -> None:
        super().__init__()
        self.criteria = args


class TerminateIfAny(MultipleCriteria):

    def _update(self, algorithm):
        return max([termination.update(algorithm) for termination in self.criteria])


class TerminateIfAll(MultipleCriteria):

    def _update(self, algorithm):
        return min([termination.update(algorithm) for termination in self.criteria])
