import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.optimize import minimize

problem = get_problem("sphere")

X = np.random.random((500, problem.n_var))

pop = Population.empty(len(X))
pop.set("X", X)
Evaluator().eval(problem, pop)

algorithm = GA(sampling=pop)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))