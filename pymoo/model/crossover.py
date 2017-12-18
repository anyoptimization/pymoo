from abc import abstractmethod

import numpy as np


class Crossover:
    def __init__(self, n_parents, n_children):
        self.n_parents = n_parents
        self.n_children = n_children

    def do(self, problem, parents):
        n_var = parents.shape[1]
        children = np.full((self.n_children, n_var), np.inf)
        return self._do(problem, parents, children)

    @abstractmethod
    def _do(self, problem, parents, children):
        pass
