import numpy as np

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("sphere")

algorithm = CMAES(x0=np.random.random(problem.n_var))

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")
