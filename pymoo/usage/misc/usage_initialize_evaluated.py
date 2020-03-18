import numpy as np

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_problem
from pymoo.model.evaluator import Evaluator
from pymoo.model.population import Population
from pymoo.optimize import minimize

problem = get_problem("sphere")

X = np.random.random((500, problem.n_var))

pop = Population(len(X))
pop.set("X", X)
Evaluator().eval(problem, pop)

algorithm = GA(sampling=pop)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))