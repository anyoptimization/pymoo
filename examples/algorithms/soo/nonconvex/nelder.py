from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.optimize import minimize
from pymoo.problems.single import Himmelblau

problem = Himmelblau()

algorithm = NelderMead()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
