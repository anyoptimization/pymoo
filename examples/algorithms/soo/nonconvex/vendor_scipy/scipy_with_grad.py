import numpy as np

from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.vendor.vendor_scipy import LBFGSB


class MySphere(Problem):

    def __init__(self, n_var=3):
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=0, xl=-100, xu=5, vtype=float, evaluation_of=["F", "dF"])

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum(x ** 2, axis=1)

        if "dF" in out:
            out["dF"] = (2 * x)[:, None, :]


problem = MySphere()

algorithm = LBFGSB(estm_gradients=False)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print(f"{algorithm.__class__.__name__}: Best solution found: X = {res.X} | F = {res.F} | CV = {res.F}")
