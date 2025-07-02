import numpy as np
from pymoo.util import default_random_state


class RouletteWheelSelection:

    def __init__(self, val, larger_is_better=True):
        super().__init__()
        if not larger_is_better:
            val = val.max() - val
        _sum = val.sum()
        self.cumulative = np.array([val[:k].sum() / _sum for k in range(1, len(val))])

    @default_random_state
    def next(self, n=None, random_state=None):
        if n is None:
            X = random_state.random((1, 1))
        else:
            X = random_state.random((n, 1))
            if n > 1:
                X.repeat(n - 1, axis=1)

        M = self.cumulative[None, :].repeat(len(X), axis=0)
        B = X >= M
        ret = B.sum(axis=1)

        if n is None:
            return ret[0]
        return ret
