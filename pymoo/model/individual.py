import copy


class Individual:

    def __init__(self, X=None, F=None, CV=None, G=None, feasible=None, **kwargs) -> None:
        self.X = X
        self.F = F
        self.CV = CV
        self.G = G
        self.feasible = feasible
        self.data = kwargs
        self.attr = set(self.__dict__.keys())

    def has(self, key):
        return key in self.attr or key in self.data

    def set(self, key, value):
        if key in self.attr:
            self.__dict__[key] = value
        else:
            self.data[key] = value
        return self

    def copy(self):
        ind = copy.copy(self)
        ind.data = self.data.copy()
        return ind

    def get(self, *keys):

        def _get(key):
            if key in self.data:
                return self.data[key]
            elif key in self.attr:
                return self.__dict__[key]
            else:
                return None

        ret = []

        for key in keys:
            ret.append(_get(key))

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)
