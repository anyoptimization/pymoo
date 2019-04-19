import numpy as np

from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymop import Problem


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=1, n_constr=1, xl=0, xu=10, type_var=np.int)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = - np.min(x * [3, 1], axis=1)
        out["G"] = x[:, 0] + x[:, 1] - 10


def repair(problem, pop, **kwargs):
    pop.set("X", np.round(pop.get("X")).astype(np.int))
    return pop


method = get_algorithm("ga",
                       pop_size=20,
                       sampling=get_sampling("real_random"),
                       crossover=get_crossover("real_sbx", prob_cross=1.0, eta_cross=5.0),
                       mutation=get_mutation("real_polynomial_mutation", eta_mut=3.0),
                       eliminate_duplicates=True,
                       func_repair=repair,
                       elimate_duplicates=True)

# execute the optimization
res = minimize(MyProblem(),
               method,
               termination=('n_gen', 200),
               )

print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)
