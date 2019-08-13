from pymoo.algorithms.so_nelder_mead import NelderMead
from pymoo.factory import get_problem, get_algorithm
from pymoo.optimize import minimize

problem = get_problem("go-xinsheyang04")

algorithm = NelderMead(n_max_restarts=100)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print(res.X)
print(res.F)
