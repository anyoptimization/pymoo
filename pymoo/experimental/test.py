import numpy as np
from pymop.factory import get_problem

from pymop.problem import Problem


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=1,
                         xl=np.array([-2, -2]), xu=np.array([2, 2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0] ** 2 + x[:, 1] ** 2
        f2 = (x[:, 0] - 1) ** 2 + x[:, 1] ** 2
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.full(len(x), 0)


problem = MyProblem()


problem = get_problem("zdt1")
from pymoo.factory import get_algorithm

method = get_algorithm("nsga2",
                      pop_size=20,
                      elimate_duplicates=False)


from pymoo.optimize import minimize
from pymoo.util import plotting

res = minimize(problem,
               method,
               termination=('n_gen', 1000),
               seed=1,
               save_history=True,
               disp=False)

plotting.plot(res.F)

