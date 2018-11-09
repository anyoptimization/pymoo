import copy


class Individual:

    def __init__(self, **kwargs) -> None:
        self.X = None
        self.F = None
        self.CV = None
        self.G = None
        self.feasible = None
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
