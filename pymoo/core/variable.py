import numpy as np


class Variable(object):

    def __init__(self, value=None, active=True, flag="default") -> None:
        super().__init__()
        self.value = value
        self.flag = flag
        self.active = active

    def sample(self, n=None):
        if n is None:
            return self._sample(1)[0]
        else:
            return self._sample(n)

    def _sample(self, n):
        pass

    def set(self, value):
        self.value = value

    def get(self, **kwargs):
        return self.value


class BoundedVariable(Variable):

    def __init__(self, value=None, bounds=(None, None), strict=None, **kwargs) -> None:
        super().__init__(value=value, **kwargs)
        self.bounds = bounds

        if strict is None:
            strict = bounds
        self.strict = strict

    @property
    def lb(self):
        return self.bounds[0]

    @property
    def ub(self):
        return self.bounds[1]


class Real(BoundedVariable):
    vtype = float

    def _sample(self, n):
        low, high = self.bounds
        return np.random.uniform(low=low, high=high, size=n)


class Integer(BoundedVariable):
    vtype = int

    def _sample(self, n):
        low, high = self.bounds
        return np.random.randint(low, high=high + 1, size=n)


class Binary(BoundedVariable):
    vtype = bool

    def _sample(self, n):
        return np.random.random(size=n) < 0.5


class Choice(Variable):
    vtype = object

    def __init__(self, value=None, options=None, all=None, **kwargs) -> None:
        super().__init__(value=value, **kwargs)
        self.options = options

        if all is None:
            all = options
        self.all = all

    def _sample(self, n):
        return np.random.choice(self.options, size=n)


def get(*args, size=None, **kwargs):
    if len(args) == 0:
        return

    ret = []
    for arg in args:
        v = arg.get(**kwargs) if isinstance(arg, Variable) else arg

        if size is not None:

            if isinstance(v, np.ndarray):
                v = np.reshape(v, size)
            else:
                v = np.full(size, v)

        ret.append(v)

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
