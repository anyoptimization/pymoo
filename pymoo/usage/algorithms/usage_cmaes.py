from pymoo.algorithms.soo.convex.nonderiv.cmaes import SimpleCMAES, CMAES
from pymoo.factory import get_problem
from pymoo.optimize import minimize
import numpy as np

problem = get_problem("sphere")

# algorithm = SimpleCMAES(normalize=True)
algorithm = CMAES(x0=np.random.random(problem.n_var), normalize=True)
# algorithm = BIPOPCMAES(restarts=4)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")
