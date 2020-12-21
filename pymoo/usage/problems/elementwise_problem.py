import autograd.numpy as anp
import numpy as np

from pymoo.model.problem import ElementwiseProblem
from pymoo.problems.autodiff import AutomaticDifferentiation
from pymoo.problems.numdiff import NumericalDifferentiation
from pymoo.problems.single import Sphere


class ElementwiseSphere(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, xl=-1, xu=1, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = ((x-0.5) ** 2).sum()


X = 5 * np.random.random((5, 2))


problem = AutomaticDifferentiation(ElementwiseSphere())
F, dF = problem.evaluate(X, return_values_of=["F", "dF"])


problem = AutomaticDifferentiation(Sphere(n_var=2))
_F, _dF = problem.evaluate(X, return_values_of=["F", "dF"])

problem = NumericalDifferentiation(Sphere(n_var=2))
__F, __dF = problem.evaluate(X, return_values_of=["F", "dF"])


print(np.column_stack([(2 * (X-0.5))[:, None], dF, _dF, __dF]))

#
# print("TNK")
#
# problem = AutomaticDifferentiation(TNK())
# _F, _dF, _dG = problem.evaluate(X, return_values_of=["F", "dF", "dG"])
#
# print(_dF, _dG)
