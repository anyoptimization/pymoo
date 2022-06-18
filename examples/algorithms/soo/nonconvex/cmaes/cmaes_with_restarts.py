import numpy as np

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.problems.single import Ackley

problem = Ackley()

algorithm = CMAES(x0=np.random.random(problem.n_var), restarts=5)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")

