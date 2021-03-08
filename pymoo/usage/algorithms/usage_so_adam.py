import numpy as np

from pymoo.algorithms.so_adam import Adam
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("rosenbrock")

X = np.random.random((1, problem.n_var))

algorithm = Adam(X)

res = minimize(problem,
               algorithm,
               ("n_evals", 100),
               seed=2,
               verbose=True)

print(res.X)
print(res.F)
