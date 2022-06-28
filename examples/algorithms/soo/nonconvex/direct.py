from pymoo.algorithms.soo.nonconvex.direct import DIRECT
from pymoo.optimize import minimize
from pymoo.problems.single import Rosenbrock

problem = Rosenbrock(n_var=2)
problem.xl -= 10

algorithm = DIRECT()

ret = minimize(problem,
               algorithm,
               ("n_evals", 5000),
               verbose=True)

