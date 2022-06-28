import numpy as np

from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.vendor.vendor_scipy import LBFGSB, TNC


class MySphere(Problem):

    def __init__(self, n_var=3):
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=0, xl=-5, xu=5, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum(np.square(x - 10), axis=1)


problem = MySphere()

algorithms = [LBFGSB(), TNC()]

for algorithm in algorithms:

    res = minimize(problem,
                   algorithm,
                   seed=1,
                   verbose=False)

    print(f"{algorithm.__class__.__name__}: Best solution found: X = {res.X} | F = {res.F} | CV = {res.F}")
