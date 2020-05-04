from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util.termination.constr_violation import ConstraintViolationToleranceTermination

problem = get_problem("g05")
algorithm = GA(pop_size=100)

res = minimize(problem,
               algorithm,
               ConstraintViolationToleranceTermination(),
               return_least_infeasible=True,
               seed=1,
               verbose=True)

print(res.CV[0])
print(res.F[0])