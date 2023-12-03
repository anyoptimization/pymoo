import numpy as np

from pymoo.core.problem import Problem, ProblemType
from pymoo.core.solution import Solution
from pymoo.core.variable import Float, VariableType


class CustomSphere(Problem):

    def __init__(self,
                 vtype: VariableType = Float(size=10, bounds=(-10, 10)),
                 constrained=False
                 ) -> None:

        ptype = ProblemType()
        if constrained:
            ptype.n_ieq_constr = 1

        super().__init__(vtype=vtype, ptype=ptype, parallel=False)
        self.constrained = True

    def _evaluate(self, x: np.ndarray, out: dict) -> None:
        out["F"][:] = 1 + np.sum(x ** 2)
        if self.constrained:
            out["G"][:] = 100 - np.sum(x ** 2)

    def fopt(self):
        return 1.0


