import abc

import numpy as np

from pymoo.util import default_random_state


class Operator:

    def __init__(self,
                 name=None,
                 vtype=None,
                 repair=None) -> None:

        super().__init__()

        if name is None:
            name = self.__class__.__name__

        self.name = name
        self.vtype = vtype
        self.repair = repair

    @default_random_state
    def do(self, problem, elem, *args, random_state, **kwargs):
        return self._do(problem, elem, *args, random_state=random_state, **kwargs)

    @abc.abstractmethod
    def _do(self, problem, elem, *args, random_state, **kwargs):
        pass

    def __call__(self, problem, elem, *args, to_numpy=False, **kwargs):
        out = self.do(problem, elem, *args, **kwargs)

        if self.vtype is not None:
            for ind in out:
                ind.X = ind.X.astype(self.vtype)

        # allow to have a built-in repair (can be useful to customize standard crossover)
        if self.repair is not None:
            self.repair.do(problem, out)

        if to_numpy:
            out = np.array([ind.X for ind in out])

        return out
