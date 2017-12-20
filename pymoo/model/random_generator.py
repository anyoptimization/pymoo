from abc import abstractmethod


class RandomGenerator:
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