from pymoo.algorithms.soo.convex.nonderiv.pattern_search import PatternSearch
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("bbob-f10-1", n_var=40)

algorithm = PatternSearch()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))