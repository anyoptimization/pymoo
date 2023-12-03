from abc import abstractmethod

import numpy as np

from pymoo.core.variable import VariableType, Variable


class ProblemType:

    def __init__(self,
                 n_obj: int = 1,
                 n_ieq_constr: int = 0,
                 n_eq_constr: int = 0):
        super().__init__()
        self.n_obj = n_obj
        self.n_ieq_constr = n_ieq_constr
        self.n_eq_constr = n_eq_constr

    @property
    def n_constr(self):
        return self.n_eq_constr + self.n_ieq_constr
    def has_constraints(self):
        return self.n_constr > 0


class Problem:

    def __init__(self,
                 vtype: VariableType = None,
                 ptype: ProblemType = ProblemType(),
                 parallel: bool = False,
                 gradient: bool = False,
                 hessian: bool = False) -> None:
        """
        This class defines an optimization problem and its corresponding properties.

        Parameters
        ----------
        vtype : VariableType
            The variable type for this problem.
        ptype : ProblemType
            The type of the problem
        parallel : bool
            Whether the problem supports evaluation in parallel.
        """
        super().__init__()
        self.vtype = vtype
        self.ptype = ptype
        self.parallel = parallel
        self.gradient = gradient
        self.hessian = hessian

    def evaluate(self, var: Variable, out: dict = None) -> dict:

        if out is None:
            out = self.default_out()

        x = var.get()
        self._evaluate(x, out)

        return out

    def default_shapes(self, n=None):
        ptype, vtype = self.ptype, self.vtype

        shapes = dict(
            F=ptype.n_obj,
            G=ptype.n_ieq_constr,
            H=ptype.n_eq_constr,
        )

        if self.gradient:
            d = dict(dF=(ptype.n_obj, vtype.size),
                     dG=(ptype.n_ieq_constr, vtype.size),
                     dH=(ptype.n_eq_constr, vtype.size)
                     )
            shapes = {**shapes, **d}

        if self.hessian:
            d = dict(ddF=(ptype.n_obj, vtype.size, vtype.size),
                     ddG=(ptype.n_ieq_constr, vtype.size, vtype.size),
                     ddH=(ptype.n_eq_constr, vtype.size, vtype.size)
                     )
            shapes = {**shapes, **d}

        if n is not None:
            shapes = {name: (n, *shape) for name, shape in shapes}

        return shapes

    def default_out(self, n=None):
        return {name: np.full(shape, np.nan) for name, shape in self.default_shapes(n=n).items()}

    @abstractmethod
    def _evaluate(self, x: np.ndarray, out: dict) -> None:
        pass
