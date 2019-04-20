from abc import abstractmethod


class Sampling:
    """

    This abstract class represents any sampling strategy that can be used to create an initial population or
    an initial search point.

    """

    @abstractmethod
    def sample(self, problem, pop, n_samples, **kwargs):
        """
        Sample new points with problem information if necessary.

        Parameters
        ----------
        problem: class
            The problem to which points should be sampled. (lower and upper bounds, discrete, binary, ...)

        n_samples: int
            Number of samples

        kwargs: class
            Any additional data that might be necessary. e.g. constants of the algorithm, ...

        Returns
        -------
        X : np.array
            Samples points in a two dimensional array

        """
        return pop
