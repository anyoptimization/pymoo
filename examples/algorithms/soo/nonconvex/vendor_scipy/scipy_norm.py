import numpy as np

from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.problems.zero_to_one import ZeroToOne
from pymoo.vendor.vendor_scipy import LBFGSB


class MySphere(Problem):

    def __init__(self, n_var=3):
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=0, xl=-100, xu=5, vtype=np.double)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum(np.square(x + 10), axis=1)


# for the local optimizer this is now a perfectly normalized problem
problem = ZeroToOne(MySphere())

algorithm = LBFGSB()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

# map the solution back to the original space
X = problem.denormalize(res.X)

print(f"{algorithm.__class__.__name__}: Best solution found: X = {X} | F = {res.F} | CV = {res.F}")
