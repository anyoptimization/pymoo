from pymoo.algorithms.so_pso import PSO
from pymoo.factory import Rastrigin
from pymoo.optimize import minimize

problem = Rastrigin()

algorithm = PSO()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))