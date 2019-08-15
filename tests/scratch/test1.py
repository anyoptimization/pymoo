from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize

import numpy as np

problem = get_problem("zdt2")

algorithm = NSGA2(pop_size=100, elimate_duplicates=True)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)

np.savetxt("performance_indicators_4.f", res.F)
np.savetxt("performance_indicators.pf", problem.pareto_front())
