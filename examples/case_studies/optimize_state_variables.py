import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

n_vars = 5
n_states = 50


class MyProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=n_vars * n_states, n_obj=1, xl=0, xu=1, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.reshape(x, (n_vars, n_states))
        out["F"] = ((x - np.linspace(0, 1, n_states)) ** 2).sum()


problem = MyProblem()

algorithm = GA()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

x = res.X
print(np.reshape(x, (n_vars, n_states)))
