from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util.termination.constr_violation import ConstraintViolationToleranceTermination

problem = get_problem("g03")
algorithm = GA(pop_size=100)

res = minimize(problem,
               algorithm,
               ConstraintViolationToleranceTermination(n_last=20, nth_gen=5),
               seed=1,
               verbose=True)

print(res.CV[0])
