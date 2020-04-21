from pymoo.algorithms.so_nelder_mead import NelderMead
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("sphere")

algorithm = NelderMead()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print(res.X)
print(res.F)
