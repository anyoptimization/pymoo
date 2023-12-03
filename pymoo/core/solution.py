import numpy as np


class Solution:

    def __init__(self, **kwargs) -> None:
        # function values
        self._var = None
        self._F, self._G, self._H = None, None, None
        self._CV = None

        # first derivative
        self._dF, self._dG, self._dH = None, None, None

        # second derivative
        self._ddF, self._ddG, self._ddH = None, None, None

        # to store additional information
        self.data = dict()

        for k, v in kwargs.items():
            if k in self.__dict__:
                self.__dict__[k] = v
            elif "_" + k in self.__dict__:
                self.__dict__["_" + k] = v
            else:
                self.data[k] = v

    @property
    def var(self):
        return self._var

    @var.setter
    def var(self, value):
        self._var = value

    @property
    def X(self):
        return self._var.get()

    @X.setter
    def X(self, value):
        self._var.set(value)

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
        GH = np.concatenate([self._G, self._H])
        if len(GH) > 0:
            return np.maximum(0.0, GH)
        else:
            return np.zeros(1)

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
    # Convenience
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
    # Functions
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


class SolutionSet(np.ndarray):

    def __new__(cls, sols=()):
        return np.array(sols).view(cls)

    def set(self, **kwargs):

        # if population is empty just return
        if self.size == 0:
            return

        # for each entry in the dictionary set it to each individual
        for key, values in kwargs.items():
            is_iterable = hasattr(values, '__len__') and not isinstance(values, str)

            if is_iterable and len(values) != len(self):
                raise Exception("Solution Set Error: Number of values and solution set size do not match.")

            for i in range(len(self)):
                val = values[i] if is_iterable else values

                # check for view and make copy to prevent memory leakage (#455)
                if isinstance(val, np.ndarray) and not val.flags["OWNDATA"]:
                    val = val.copy()

                self[i].set(key, val)

        return self

    def get(self, *args, to_numpy=True, **kwargs):

        val = {}
        for c in args:
            val[c] = []

        # for each individual
        for i in range(len(self)):

            # for each argument
            for c in args:
                val[c].append(self[i].get(c, **kwargs))

        # convert the results to a list
        res = [val[c] for c in args]

        # to numpy array if desired - default true
        if to_numpy:
            res = [np.array(e) for e in res]

        # return as tuple or single value
        if len(args) == 1:
            return res[0]
        else:
            return tuple(res)

    @property
    def F(self):
        return self.get("F")

    @F.setter
    def F(self, value):
        self._F = value


def merge(a: object, b: object) -> SolutionSet:
    if a is None:
        return b
    elif b is None:
        return a
    else:
        a = SolutionSet([a]) if isinstance(a, Solution) else a
        b = SolutionSet([b]) if isinstance(b, Solution) else b
        return np.concatenate([a, b]).view(SolutionSet)


