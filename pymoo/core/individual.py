import copy


class Individual:

    def __init__(self,
                 X=None, F=None, G=None,
                 dF=None, dG=None,
                 ddF=None, ddG=None,
                 CV=None, feasible=None,
                 **kwargs) -> None:

        # design variables
        self.X = X

        # objectives and constraint values
        self.F = F
        self.G = G

        # first order derivation
        self.dF = dF
        self.dG = dG

        # second order derivation
        self.ddF = ddF
        self.ddG = ddG

        # constraint violation and feasibility flag
        self.CV = CV
        self.feasible = feasible

        # a set storing what has been evaluated
        self.evaluated = set()

        # additional data to be set
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

    def set_by_dict(self, **kwargs):
        for k, v in kwargs.items():
            self.set(k, v)

    def copy(self, deep=False):
        ind = copy.copy(self)
        ind.data = copy.copy(self.data) if not deep else copy.deepcopy(self.data)
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

