from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.optimize import minimize
from pymoo.problems.single import G1
from pymoo.visualization.scatter import Scatter


problem = G1()

algorithm = AdaptiveEpsilonConstraintHandling(GA())

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               copy_algorithm=False,
               save_history=False,
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))
