from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.termination.cv import ConstraintViolationTermination
from pymoo.termination.robust import RobustTermination

problem = get_problem("g5")
algorithm = GA(pop_size=100)

res = minimize(problem,
               algorithm,
               RobustTermination(ConstraintViolationTermination(), period=30),
               return_least_infeasible=True,
               seed=1,
               verbose=True)

print(res.CV[0])
print(res.F[0])
