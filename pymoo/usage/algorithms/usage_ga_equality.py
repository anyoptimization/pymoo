import numpy as np

from pymoo.model.problem import Problem


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=3,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array([-2, -2, -2]),
                         xu=np.array([2, 2, 2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0] ** 2 + x[:, 1] ** 2
        f2 = (x[:, 0] - 1) ** 2 + x[:, 1] ** 2
        g1 = (x[:, 0] + x[:, 2] - 2) ** 2 - 1e-5

        out["F"] = np.column_stack([f1, f2])
        out["G"] = g1


from pymoo.model.repair import Repair


class MyRepair(Repair):

    def _do(self, problem, pop, **kwargs):
        for k in range(len(pop)):
            x = pop[k].X
            if np.random.random() < 0.5:
                x[2] = 2 - x[0]
            else:
                x[0] = 2 - x[2]
        return pop


from pymoo.algorithms.nsga2 import NSGA2

algorithm = NSGA2(pop_size=100, repair=MyRepair(), eliminate_duplicates=True)

from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

res = minimize(MyProblem(),
               algorithm,
               ('n_gen', 20),
               seed=1,
               verbose=True)

plot = Scatter()
plot.add(res.F, color="red")
plot.show()
