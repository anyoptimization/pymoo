from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.termination.robust import RobustTermination
from pymoo.termination.xtol import DesignSpaceTermination

problem = get_problem("zdt1")
algorithm = NSGA2(pop_size=100)
termination = RobustTermination(DesignSpaceTermination(tol=0.01), period=30)

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               verbose=True)

print(res.algorithm.n_gen)
