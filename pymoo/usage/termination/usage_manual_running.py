from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util.running_metric import RunningMetric

problem = get_problem("zdt3")
algorithm = NSGA2(pop_size=100, callback=RunningMetric(5))

if False:
    minimize(problem,
             algorithm,
             pf=False,
             seed=1,
             verbose=True)
