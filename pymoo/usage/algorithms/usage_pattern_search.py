from pymoo.algorithms.so_pattern_search import PatternSearch
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("ackley", n_var=30)

algorithm = PatternSearch()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))