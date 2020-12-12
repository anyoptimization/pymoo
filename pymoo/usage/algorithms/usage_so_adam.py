import numpy as np

from pymoo.algorithms.soo.convex.deriv.adam import Adam
from pymoo.factory import Rosenbrock
from pymoo.optimize import minimize

problem = Rosenbrock()

X = np.random.random((1, problem.n_var))

algorithm = Adam(X)

res = minimize(problem,
               algorithm,
               ("n_evals", 100),
               seed=2,
               verbose=True)

print(res.X)
print(res.F)
