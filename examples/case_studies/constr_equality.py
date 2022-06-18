import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=3,
                         n_obj=2,
                         n_ieq_constr=1,
                         xl=np.array([-2, -2, -2]),
                         xu=np.array([2, 2, 2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0] ** 2 + x[:, 1] ** 2
        f2 = (x[:, 0] - 1) ** 2 + x[:, 1] ** 2
        g1 = (x[:, 0] + x[:, 2] - 2) ** 2 - 1e-5

        out["F"] = np.column_stack([f1, f2])
        out["G"] = g1


class MyRepair(Repair):

    def _do(self, problem, X, **kwargs):
        for k in range(len(X)):
            if np.random.random() < 0.5:
                X[k, 2] = 2 - X[k, 0]
            else:
                X[k, 0] = 2 - X[k, 2]
        return X


algorithm = NSGA2(pop_size=100, repair=MyRepair(), eliminate_duplicates=True)

res = minimize(MyProblem(),
               algorithm,
               ('n_gen', 20),
               seed=1,
               verbose=True)

plot = Scatter()
plot.add(res.F, color="red")
plot.show()
