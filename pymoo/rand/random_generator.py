from abc import abstractmethod


class RandomGenerator:
    """
    Implementation of a random generator used for all algorithm. This is just the base class which needs to
    be inherited from.
    """

    @abstractmethod
    def seed(self, s):
        pass

    @abstractmethod
    def perm(self, size):
        pass

    @abstractmethod
    def rand(self, size=None):
        pass

    @abstractmethod
    def randint(self, low, high, size=None):
        pass