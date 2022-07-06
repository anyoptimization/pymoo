from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.optimize import minimize
from pymoo.problems.single import G1

problem = G1()

algorithm = AdaptiveEpsilonConstraintHandling(DE(), perc_eps_until=0.5)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))