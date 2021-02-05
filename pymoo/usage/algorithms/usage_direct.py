from pymoo.algorithms.soo.nonconvex.direct import DIRECT
from pymoo.optimize import minimize
from pymoo.problems.single import Ackley

problem = Ackley(n_var=5)
problem.xl -= 10

algorithm = DIRECT()

ret = minimize(problem,
               algorithm,
               ("n_evals", 2000),
               verbose=True)

