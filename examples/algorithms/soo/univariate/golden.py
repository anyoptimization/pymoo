from pymoo.algorithms.soo.univariate.golden import GoldenSectionSearch
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere

problem = Sphere(n_var=1)

algorithm = GoldenSectionSearch()

res = minimize(problem, algorithm)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
