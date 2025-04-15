from copy import deepcopy
from typing import Any

from pymoo.core.algorithm import Algorithm


class MetaAlgorithm:

    def __new__(cls, algorithm) -> Any:




        return algorithm


    def __init__(self, algorithm) -> None:
        super().__init__()
        self.algorithm = algorithm

    def __getattr__(self, attr):
        print(attr)
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.__dict__["algorithm"], attr)


class MetaAlgorithm2(Algorithm):

    def __init__(self,
                 algorithm,
                 copy=False,
                 ):
        super().__init__()

        # if the algorithm object should be copied to keep the original one unmodified
        if copy:
            algorithm = deepcopy(algorithm)

        self.algorithm = algorithm

    def _copy_from_orig(self):
        for k, v in self.algorithm.__dict__.items():
            if k not in ["opt", "display"]:
                self.__dict__[k] = v

    def setup(self, *args, **kwargs):
        self.algorithm.setup(*args, **kwargs)
        self._copy_from_orig()

        self.display = self.algorithm.display
        self.algorithm.display = None

        self._setup(*args, **kwargs)

        return self

    def _infill(self):
        ret = self.algorithm.infill()
        self._copy_from_orig()
        return ret

    def _initialize_infill(self):
        return self._infill()

    def _initialize_advance(self, infills=None, **kwargs):
        self.algorithm.advance(infills=infills, **kwargs)
        self._copy_from_orig()

    def _advance(self, infills=None, **kwargs):
        self.algorithm.advance(infills=infills, **kwargs)
        self._copy_from_orig()

    def advance(self, infills=None, **kwargs):
        super().advance(infills, **kwargs)
        self._copy_from_orig()

    def _set_optimum(self):
        self.opt = self.algorithm.opt
