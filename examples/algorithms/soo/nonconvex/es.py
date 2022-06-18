from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("ackley", n_var=10)

algorithm = ES(200)

res = minimize(problem,
               algorithm,
               ("n_gen", 1000),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
