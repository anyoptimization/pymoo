from pymoo.algorithms.so_golden import GoldenSectionSearch
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere

problem = Sphere(n_var=1)

algorithm = GoldenSectionSearch()

res = minimize(problem, algorithm, ("n_iter", 30))

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
