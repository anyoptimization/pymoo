# START getting_start_problem
import numpy as np
from pymoo.model.problem import Problem


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=1,
                         xl=np.array([-2,-2]), xu=np.array([2,2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:,0]**2 + x[:,1]**2
        f2 = (x[:,0]-1)**2 + x[:,1]**2
        out["F"] = np.column_stack([f1, f2])
        out["G"] = - (x[:,0]**2 - x[:,0] + 3/16)

problem = MyProblem()
# END getting_start_problem


# START getting_start_method
from pymoo.algorithms.nsga2 import nsga2

method = nsga2(pop_size=100,
               elimate_duplicates=True)
# END getting_start_method


# START getting_start_minimize
from pymoo.optimize import minimize

res = minimize(problem,
               method,
               ('n_gen', 200),
               seedp=1,
               verbose=False)
# END getting_start_minimize

