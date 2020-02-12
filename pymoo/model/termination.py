class Termination:

    def __init__(self) -> None:
        """
        Base class for the implementation of a termination criterion for an algorithm.
        """
        super().__init__()

        # the algorithm can be forced to terminate by setting this attribute to true
        self.force_termination = False

    def do_continue(self, algorithm):
        """

        Whenever the algorithm objects wants to know whether it should continue or not it simply
        asks the termination criterion for it.

        Parameters
        ----------
        algorithm : class
            The algorithm object that is asking if it has terminated or not.

        Returns
        -------
        do_continue : bool
            Whether the algorithm has terminated or not.

        """

        if self.force_termination:
            return False
        else:
            return self._do_continue(algorithm)

    # the concrete implementation of the algorithm
    def _do_continue(self, algorithm, **kwargs):
        pass

    def has_terminated(self, algorithm):
        """
        Instead of asking if the algorithm should continue it can also ask if it has terminated.
        (just negates the continue method.)
        """
        return not self.do_continue(algorithm)
