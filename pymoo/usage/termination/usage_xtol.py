from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util.termination.x_tol import DesignSpaceToleranceTermination

problem = get_problem("zdt3")
algorithm = NSGA2(pop_size=100)
termination = DesignSpaceToleranceTermination(tol=0.0025, n_last=20)

res = minimize(problem,
               algorithm,
               termination,
               pf=problem.pareto_front(),
               seed=1,
               verbose=False)

print(res.algorithm.n_gen)
