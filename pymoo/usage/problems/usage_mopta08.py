from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.problems.single.mopta08 import MOPTA08

problem = MOPTA08("/Users/blankjul/Downloads/mopta_fortran_unix/exec")
algorithm = GA(n_offsprings=20, return_least_infeasible=True)

res = minimize(problem,
               algorithm,
               verbose=True
               )

print(res.F, res.F)
