import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.model.evaluator import Evaluator
from pymoo.model.population import Population
from pymoo.optimize import minimize

problem = get_problem("zdt2")

# create initial data and set to the population object
X = np.random.random((300, problem.n_var))
pop = Population.new("X", X)
Evaluator().eval(problem, pop)

algorithm = NSGA2(pop_size=100, sampling=pop)

minimize(problem,
         algorithm,
         ('n_gen', 10),
         seed=1,
         verbose=True)