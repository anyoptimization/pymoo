import autograd.numpy as anp
import numpy as np

from pymoo.algorithms.moead import MOEAD
from pymoo.factory import get_reference_directions
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=3,
                         n_obj=3,
                         n_constr=0,
                         xl=anp.array([0.0, 0.0, 0.0]),
                         xu=anp.array([2.0, 2.0, 2.0]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 10 + (x[:, 0] - 2)**2 + x[:, 1]**4 + 5 * x[:, 2]
        f2 = x[:, 0]**2 + (x[:, 1] - 1)**2 + 4 * (x[:, 2] - 2)**3
        f3 = 2 * (x[:, 0] + 2) + (x[:, 1] - 2)**3 + (x[:, 2] - 1)**2

        out["F"] = anp.column_stack([f1, f2, f3])


problem = MyProblem()

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=16)
algorithm = MOEAD(ref_dirs)

ret = minimize(problem, algorithm, verbose=True)

Scatter().add(ret.F).show()

np.savetxt("deb_sample.x", ret.X)
np.savetxt("deb_sample.f", ret.F)


print(np.min(ret.F, axis=0))
print(np.max(ret.F, axis=0))