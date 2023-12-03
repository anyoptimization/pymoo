import numpy as np

from pymoo.core.solution import Solution
from pymoo.util import is_iterable


class VariableType:

    def __init__(self,
                 dtype: type = object,
                 size: int = None,
                 bounds: tuple = None):
        super().__init__()
        self.dtype = dtype
        self.size = size

        if bounds is not None:
            low, high = bounds

            if self.size is not None:
                low = np.array(low) if is_iterable(low) else np.full(self.size, low)
                high = np.array(high) if is_iterable(high) else np.full(self.size, high)
                bounds = (low, high)

                assert len(low) == size and len(high) == size

        self.bounds = bounds

    @property
    def low(self):
        if self.has_bounds():
            return self.bounds[0]

    @property
    def high(self):
        if self.has_bounds():
            return self.bounds[1]

    def has_bounds(self):
        return self.bounds is not None

    def norm(self):
        if self.has_bounds():
            low, high = self.bounds
            return high - low
        else:
            return 1.0

    def new(self, value: object) -> 'Variable':
        return Variable(self, value)

    def random(self):
        pass

    def full(self, v):
        pass


class Float(VariableType):

    def __init__(self, bounds=(0, 1), **kwargs):
        super().__init__(dtype=float, bounds=bounds, **kwargs)

    def random(self):
        x = np.random.uniform(size=self.size)

        if self.bounds:
            low, high = self.bounds
            x = low + x * (high - low)

        return self.new(x)

    def full(self, v):
        return self.new(np.full(self.size, v))


class Integer(VariableType):

    def __init__(self, bounds=(0, 1), **kwargs):
        super().__init__(dtype=int, bounds=bounds, **kwargs)

    def random(self):
        x = np.random.randint(self.low, high=self.high + 1, size=self.size)
        return self.new(x)


class Binary(VariableType):

    def __init__(self, **kwargs):
        super().__init__(dtype=bool, **kwargs)

    def random(self):
        x = np.random.uniform(size=self.size) < 0.5
        return self.new(x)


class Mixed(VariableType):

    def __init__(self, vars):
        super().__init__(dtype=object, size=len(vars), bounds=None)
        self.vars = vars

    def random(self):
        x = [var.random().get() for var in self.vars]
        return self.new(x)


class Variable:

    def __init__(self, vtype: VariableType, value: object):
        super().__init__()
        self.vtype = vtype
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value

    def solution(self) -> 'Solution':
        return Solution(var=self)
