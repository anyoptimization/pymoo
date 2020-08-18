from pymoo.algorithms.so_cmaes import CMAES
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("sphere")

algorithm = CMAES()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")