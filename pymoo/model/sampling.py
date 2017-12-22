from abc import abstractmethod


class Sampling:
    """

    This abstract class represents any sampling strategy that can be used
    to create an initial population or just an initial point

    """

    @abstractmethod
    def sample(self, problem, n_samples, data=None):
        """
        Sample new points according to the problem.


        Parameters
        ----------
        problem: class
            The problem to which points should be sampled. (lower and upper bounds, discrete, binary, ...)

        n_samples: int
            Number of points that should be created

        data: class
            Any additional data that might be necessary. e.g. constants of the algorithm, ...

        Returns
        -------

        """
        pass
