from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("g01")

algorithm = SRES()

res = minimize(problem,
               algorithm,
               ("n_gen", 875),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))
