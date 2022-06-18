from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.constraints.from_bounds import ConstraintsFromBounds
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere

problem = Sphere()

problem = ConstraintsFromBounds(problem, remove_bonds=False)

algorithm = DE()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))