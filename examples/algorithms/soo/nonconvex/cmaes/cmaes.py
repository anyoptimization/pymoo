import numpy as np

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.problems.single import Himmelblau

problem = Himmelblau()

algorithm = CMAES(x0=np.random.random(problem.n_var))

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")

