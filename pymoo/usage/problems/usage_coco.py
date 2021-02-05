from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("bbob-f01-1", n_var=10)

res = minimize(problem,
               DE(),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
