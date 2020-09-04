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


class eIndividual:

    def __init__(self, **kwargs) -> None:
        kwargs = {**kwargs, **dict(X=None, F=None, CV=None, G=None, feasible=None)}
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def has(self, key):
        return key in self.__dict__

    def set(self, key, val):
        self.__dict__[key] = val

    def get(self, *keys):
        if len(keys) == 1:
            return self.__dict__.get(keys[0])
        else:
            return tuple([self.__dict__.get(key) for key in keys])

    def copy(self, deep=False):
        ind = copy.copy(self)
        ind.data = copy.copy(self.__dict__) if not deep else copy.deepcopy(self.__dict__)
        return ind


        if not deep:
            d = dict(self.__dict__)
        else:
            d = copy.deepcopy(self.__dict__)
        ind = Individual(**d)
        return ind

    # @property
    # def F(self):
    #     attr = "F"
    #     if attr in self.__dict__:
    #         return self.__dict__[attr]
    #     else:
    #         return None

    # Gets called when the item is not found via __getattribute__
    # def __getattr__(self, item):
    #     return super(Individual, self).__setattr__(item, 'orphan')

    def __getattr__(self, val):
        return self.__dict__.get(val)

    # def __setitem__(self, key, value):
    #     self.__dict__[key] = value
    #
    # def __getitem__(self, key):
    #     return self.__dict__.get(key)

    # def __getattr__(self, attr):
    #
    #     if attr == "F":
    #         if attr in self.__dict__:
    #             return self.__dict__[attr]
    #         else:
    #             return None
    #
    #     if attr in self.__dict__:
    #         return self.__dict__[attr]
    #
    #
    #
    def __setattr__(self, key, value):
        self.__dict__[key] = value
