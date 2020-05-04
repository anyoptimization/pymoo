import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize


class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([-2.0, -2.0, -2.0]),
                         xu=np.array([2.0, 2.0, 2.0]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 1 - np.exp(-((x - 1 / np.sqrt(self.n_var)) ** 2).sum(axis=1))
        f2 = 1 - np.exp(-((x + 1 / np.sqrt(self.n_var)) ** 2).sum(axis=1))
        out["F"] = np.column_stack([f1, f2])


problem = MyProblem()

algorithm = NSGA2(pop_size=100, eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)
