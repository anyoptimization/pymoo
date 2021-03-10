from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
from pymoo.factory import Himmelblau
from pymoo.optimize import minimize


problem = Himmelblau()

algorithm = PatternSearch()

res = minimize(problem,
               algorithm,
               verbose=False,
               seed=1)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
