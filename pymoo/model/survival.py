from abc import abstractmethod


class Survival:

    def __init__(self):
        pass

    def do(self, pop, n_survive):
        return self._do(pop, n_survive)

    @abstractmethod
    def _do(self, pop, n_survive):
        pass
