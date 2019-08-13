import copy


class Individual:

    def __init__(self, X=None, F=None, CV=None, G=None, feasible=None, **kwargs) -> None:
        self.X = X
        self.F = F
        self.CV = CV
        self.G = G
        self.feasible = feasible
        self.data = kwargs

    def set(self, key, value):
        self.data[key] = value

    def copy(self):
        ind = copy.copy(self)
        ind.data = self.data.copy()
        return ind

    def get(self, key):
        if key in self.data:
            return self.data[key]
        else:
            return None
