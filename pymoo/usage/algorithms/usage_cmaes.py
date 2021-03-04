import numpy as np

from pymoo.algorithms.soo.convex.nonderiv.cmaes import CMAES
from pymoo.factory import Sphere
from pymoo.optimize import minimize

problem = Sphere()

algorithm = CMAES(x0=np.random.random(problem.n_var))

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")
