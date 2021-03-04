from pymoo.algorithms.soo.convex.deriv.gd import GradientDescent
from pymoo.factory import Sphere
from pymoo.operators.sampling.random_sampling import random
from pymoo.optimize import minimize
from pymoo.problems.autodiff import AutomaticDifferentiation


problem = AutomaticDifferentiation(Sphere())
problem.xl += 10
problem.xu += 11

X = random(problem, 1)

algorithm = GradientDescent(strict_bounds=False)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print(res.X)
print(res.F)
