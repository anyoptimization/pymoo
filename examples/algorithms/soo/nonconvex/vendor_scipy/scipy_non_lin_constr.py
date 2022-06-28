import numpy as np

from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.vendor.vendor_scipy import TrustConstr, SLSQP


class MySphere(Problem):

    def __init__(self, n_var=3):
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=1, xl=-5, xu=5, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum(np.square(x - 10), axis=1)
        out["G"] = np.linalg.norm(0.3 - x, axis=1) - 0.3


problem = MySphere()

algorithms = [SLSQP(), TrustConstr()]

for algorithm in algorithms:

    res = minimize(problem,
                   algorithm,
                   seed=1,
                   verbose=False)

    print(f"{algorithm.__class__.__name__}: Best solution found: X = {res.X} | F = {res.F} | CV = {res.F}")
