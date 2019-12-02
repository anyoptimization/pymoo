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

    def copy(self):
        ind = copy.copy(self)
        ind.data = self.data.copy()
        return ind

    def get(self, keys):
        if keys in self.data:
            return self.data[keys]
        elif keys in self.attr:
            return self.__dict__[keys]
        else:
            return None
