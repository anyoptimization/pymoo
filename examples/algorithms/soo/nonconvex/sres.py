from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("g1")

algorithm = SRES()

res = minimize(problem,
               algorithm,
               ("n_gen", 300),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))
