from pymoo.algorithms.soo.univariate.quadr_interp import QuadraticInterpolationSearch
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere

problem = Sphere(n_var=1)

algorithm = QuadraticInterpolationSearch()

res = minimize(problem, algorithm)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
