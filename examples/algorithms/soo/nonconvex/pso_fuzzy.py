from pymoo.algorithms.soo.nonconvex.pso_fuzzy import PSOFuzzy

from pymoo.optimize import minimize
from pymoo.problems.single import Ackley

problem = Ackley(n_var=10)

algorithm = PSOFuzzy(pop_size=25)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
