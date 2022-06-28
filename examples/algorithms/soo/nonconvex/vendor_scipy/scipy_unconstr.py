import numpy as np
import numpy as np

from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.vendor.vendor_scipy import CG, BFGS, Powell, NelderMead


class MySphere(Problem):

    def __init__(self, n_var=3):
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=0, xl=None, xu=None, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum(np.square(x + 10), axis=1)


problem = MySphere()

x0 = np.random.random(problem.n_var)


algorithms = [NelderMead(x0=x0), CG(x0=x0), BFGS(x0=x0), Powell(x0=x0)]

for algorithm in algorithms:

    res = minimize(problem,
                   algorithm,
                   seed=1,
                   verbose=False)

    print(f"{algorithm.__class__.__name__}: Best solution found: X = {res.X} | F = {res.F} | CV = {res.F}")
