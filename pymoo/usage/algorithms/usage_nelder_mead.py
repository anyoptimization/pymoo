from pymoo.algorithms.so_nelder_mead import NelderMead
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("sphere")

algorithm = NelderMead()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))