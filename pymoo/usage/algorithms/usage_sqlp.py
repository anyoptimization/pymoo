import numpy as np

from pymoo.algorithms.so_sqlp import SQLP
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.problems.single import Himmelblau, Rastrigin

problem = Rastrigin(n_var=10)
problem = get_problem("g01")

algorithm = SQLP(x0=np.zeros(problem.n_var) + 0.6)

ret = minimize(problem,
               algorithm,
               seed=2,
               save_history=True,
               verbose=True)

print(ret.X)
print("Optimal:", problem.pareto_front()[0])
print(ret.F)
print(ret.pop.get("X"))
print(ret.pop.get("F"))


