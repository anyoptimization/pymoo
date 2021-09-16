from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.factory import Problem, get_problem
from pymoo.optimize import minimize

import numpy as np


class MyProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=0, n_eq_constr=1, xl=0, xu=1, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (x ** 2).sum(axis=1)

        h1 = x[:, 1] - x[:, 0]
        h2 = x[:, 1] + x[:, 0] - 1
        out["H"] = np.column_stack([h1, h2])


problem = MyProblem()

algorithm = ISRES(n_offsprings=200, rule=1.0 / 7.0, gamma=0.85, alpha=0.2)

for k in range(1, 25):
    label = "g%02d" % k

    problem = get_problem(label)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 1000),
                   seed=1,
                   verbose=False)

    print(label, problem.pareto_front(), res.F, res.X)
