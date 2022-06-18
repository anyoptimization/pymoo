from pymoo.algorithms.soo.nonconvex.cmaes import SimpleCMAES
from pymoo.optimize import minimize
from pymoo.problems.single import Ackley

problem = Ackley()

algorithm = SimpleCMAES()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")

