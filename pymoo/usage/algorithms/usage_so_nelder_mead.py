from pymoo.algorithms.so_nelder_mead import NelderMead
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("sphere")

algorithm = NelderMead(n_max_restarts=10)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print(res.X)
print(res.F)
