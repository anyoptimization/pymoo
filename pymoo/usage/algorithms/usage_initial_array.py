import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("zdt2")

X = np.random.random((300, problem.n_var))

algorithm = NSGA2(pop_size=100, sampling=X)

minimize(problem,
         algorithm,
         ('n_gen', 10),
         seed=1,
         verbose=True)