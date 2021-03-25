from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("g01")

algorithm = ISRES()

res = minimize(problem,
               algorithm,
               ("n_gen", 875),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
