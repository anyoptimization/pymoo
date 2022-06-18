from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.termination.ftol import MultiObjectiveSpaceTermination

from pymoo.termination.robust import RobustTermination

problem = get_problem("zdt3")
algorithm = NSGA2(pop_size=100)

termination = RobustTermination(MultiObjectiveSpaceTermination(tol=0.025), period=30)

res = minimize(problem,
               algorithm,
               termination,
               pf=True,
               seed=1,
               verbose=False)

print("Generations", res.algorithm.n_gen)
