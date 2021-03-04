from pymoo.algorithms.soo.convex.deriv.newton import NewtonMethod
from pymoo.factory import Himmelblau
from pymoo.optimize import minimize
from pymoo.problems.autodiff import AutomaticDifferentiation

problem = AutomaticDifferentiation(Himmelblau())

algorithm = NewtonMethod()

res = minimize(problem,
               algorithm,
               seed=2,
               verbose=True)

print(res.X)
print(res.F)
