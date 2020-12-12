import numpy as np

from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs.optimizer import GradientDescent

problem = get_problem("sphere")

X = np.random.random(problem.n_var)

algorithm = GradientDescent(X)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print(res.X)
print(res.F)
