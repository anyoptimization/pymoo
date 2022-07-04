import copy

import numpy as np


def default_config():
    return dict(cache=True,
                cv_eps=0.0,
                cv_ieq=dict(scale=None, eps=0.0, pow=None, func=np.sum),
                cv_eq=dict(scale=None, eps=1e-4, pow=None, func=np.sum)
                )


class Individual:
    default_config = default_config

    def __init__(self, config=None, **kwargs) -> None:

        self._X, self._F, self._G, self._H, self._dF, self._dG, self._dH = None, None, None, None, None, None, None
        self._ddF, self._ddG, self._ddH, self._CV, = None, None, None, None
        self.evaluated = None

        # initialize all the local variables
        self.reset()

        # a local storage for data
        self.data = {}

        # the config for this individual
        if config is None:
            config = Individual.default_config()
        self.config = config

        for k, v in kwargs.items():
            if k in self.__dict__:
                self.__dict__[k] = v
            elif "_" + k in self.__dict__:
                self.__dict__["_" + k] = v
            else:
                self.data[k] = v

    def reset(self, data=True):

        empty = np.array([])

        # design variables
        self._X = empty

        # objectives and constraint values
        self._F = empty
        self._G = empty
        self._H = empty

        # first order derivation
        self._dF = empty
        self._dG = empty
        self._dH = empty

        # second order derivation
        self._ddF = empty
        self._ddG = empty
        self._ddH = empty

        # if the constraint violation value to be used
        self._CV = None

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
        config = self.config
        cache = config["cache"]

        if cache and self._CV is not None:
            return self._CV
        else:
            self._CV = np.array([calc_cv(G=self.G, H=self.H, config=config)])
            return self._CV

    @CV.setter
    def CV(self, value):
        self._CV = value

    @property
    def FEAS(self):
        eps = self.config.get("cv_eps", 0.0)
        return self.CV <= eps

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

    @ddG.setter
    def ddG(self, value):
        self._ddG = value

    @property
    def ddH(self):
        return self._ddH

    @ddH.setter
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
        if self.CV is None:
            return None
        else:
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


def calc_cv(G=None, H=None, config=None):
    if G is None:
        G = np.array([])

    if H is None:
        H = np.array([])

    if config is None:
        config = Individual.default_config()

    if G is None:
        ieq_cv = [0.0]
    elif G.ndim == 1:
        ieq_cv = constr_to_cv(G, **config["cv_ieq"])
    else:
        ieq_cv = [constr_to_cv(g, **config["cv_ieq"]) for g in G]

    if H is None:
        eq_cv = [0.0]
    elif H.ndim == 1:
        eq_cv = constr_to_cv(np.abs(H), **config["cv_eq"])
    else:
        eq_cv = [constr_to_cv(np.abs(h), **config["cv_eq"]) for h in H]

    return np.array(ieq_cv) + np.array(eq_cv)


def constr_to_cv(c, eps=0.0, scale=None, pow=None, func=np.mean):
    if c is None or len(c) == 0:
        return 0.0

    # subtract eps to allow some violation and then zero out all values less than zero
    c = np.maximum(0.0, c - eps)

    # apply init_simplex_scale if necessary
    if scale is not None:
        c = c / scale

    # if a pow factor has been provided
    if pow is not None:
        c = c ** pow

    return func(c)
