from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.optimize import minimize
from pymoo.problems import get_problem

problem = get_problem("ackley", n_var=30)

algorithm = G3PCX()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
