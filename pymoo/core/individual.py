import copy


class Individual:

    def __init__(self, **kwargs) -> None:

        self._X, self._F, self._G, self._H, self._dF, self._dG, self._dH = None, None, None, None, None, None, None
        self._ddF, self._ddG, self._ddH, self._CV, self.evaluated = None, None, None, None, None

        # initialize all the local variables
        self.reset()

        # a local storage for data
        self.data = {}

        for k, v in kwargs.items():
            if k in self.__dict__:
                self.__dict__[k] = v
            elif "_" + k in self.__dict__:
                self.__dict__["_" + k] = v
            else:
                self.data[k] = v

    def reset(self, data=True):

        # design variables
        self._X = []

        # objectives and constraint values
        self._F = []
        self._G = []
        self._H = []

        # first order derivation
        self._dF = []
        self._dG = []
        self._dH = []

        # second order derivation
        self._ddF = []
        self._ddG = []
        self._ddH = []

        # if the constraint violation value to be used
        self._CV = []

        if data:
            self.data = {}

        # a set storing what has been evaluated
        self.evaluated = set()

    def has(self, key):
        return hasattr(self.__class__, key) or key in self.data

    # -------------------------------------------------------
    # Values
    # -------------------------------------------------------

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        self._F = value

    @property
    def G(self):
        return self._G

    @G.setter
    def G(self, value):
        self._G = value

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, value):
        self._H = value

    @property
    def CV(self):
        return self._CV

    @CV.setter
    def CV(self, value):
        self._CV = value

    @property
    def FEAS(self):
        return self.CV <= 0.0

    # -------------------------------------------------------
    # Gradients
    # -------------------------------------------------------

    @property
    def dF(self):
        return self._dF

    @dF.setter
    def dF(self, value):
        self._dF = value

    @property
    def dG(self):
        return self._dG

    @dG.setter
    def dG(self, value):
        self._dG = value

    @property
    def dH(self):
        return self._dH

    @dH.setter
    def dH(self, value):
        self._dH = value

    # -------------------------------------------------------
    # Hessians
    # -------------------------------------------------------

    @property
    def ddF(self):
        return self._ddF

    @ddF.setter
    def ddF(self, value):
        self._ddF = value

    @property
    def ddG(self):
        return self._ddG

    @ddF.setter
    def ddG(self, value):
        self._ddG = value

    @property
    def ddH(self):
        return self._ddH

    @ddF.setter
    def ddH(self, value):
        self._ddH = value

    # -------------------------------------------------------
    # Convenience (value instead of array)
    # -------------------------------------------------------

    @property
    def x(self):
        return self.X

    @property
    def f(self):
        return self.F[0]

    @property
    def cv(self):
        return self.CV[0]

    @property
    def feas(self):
        return self.FEAS[0]

    # -------------------------------------------------------
    # Deprecated
    # -------------------------------------------------------

    @property
    def feasible(self):
        return self.FEAS

    # -------------------------------------------------------
    # Other Functions
    # -------------------------------------------------------

    def set_by_dict(self, **kwargs):
        for k, v in kwargs.items():
            self.set(k, v)

    def set(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.data[key] = value
        return self

    def get(self, *keys):
        ret = []

        for key in keys:
            if hasattr(self, key):
                v = getattr(self, key)
            elif key in self.data:
                v = self.data[key]
            else:
                v = None

            ret.append(v)

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    def duplicate(self, key, new_key):
        self.set(new_key, self.get(key))

    def new(self):
        return self.__class__()

    def copy(self, other=None, deep=True):
        obj = self.new()

        # if not provided just copy yourself
        if other is None:
            other = self

        # the data the new object needs to have
        D = other.__dict__

        # if it should be a deep copy do it
        if deep:
            D = copy.deepcopy(D)

        for k, v in D.items():
            obj.__dict__[k] = v

        return obj
