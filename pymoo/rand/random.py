import numpy as np

from pymoo.configuration import Configuration


# use a singleton class to set the random generator globally
class Singleton:
    __instance = None

    @staticmethod
    def get_instance():
        if Singleton.__instance is None:
            Singleton.__instance = Configuration.rand
        return Singleton.__instance


def seed(s):
    """
    Set the random seed of the current random instance that is used.

    Parameters
    ----------
    s : int
        seed

    """
    Singleton.get_instance().seed(s)


def perm(n):
    """
    Returns a random permutation with length n.

    Parameters
    ----------
    n : int
        Length of the permutation array.

    Returns
    -------
    p : numpy.array
        Permutation array with type integer.

    """
    return Singleton.get_instance().perm(n).astype(np.int)


def random(size=None):
    """
    If size is None returns one random float [0.0,1.0), other an numpy array with the predefined size.

    Parameters
    ----------
    size : tuple
        (Optional): Shape of the numpy array

    """
    return Singleton.get_instance().random(size=size)


def randint(low, high=None, size=None):
    """
    Return random integers from low (inclusive) to high (exclusive).

    Return random integers from the “discrete uniform” distribution in the “half-open” interval [low, high).
    If high is None (the default), then results are from [0, low).

    """
    if high is None:
        return Singleton.get_instance().randint(0, high=low, size=size)
    else:
        return Singleton.get_instance().randint(low, high=high, size=size)


def choice(a):
    """
    Select randomly an element from a list.
    """
    return Singleton.get_instance().choice(a)


def shuffle(a):
    """
    Shuffle a given array using the permutation random function
    """
    return Singleton.get_instance().shuffle(a)
