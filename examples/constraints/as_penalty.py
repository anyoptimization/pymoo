from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.constraints.as_penalty import ConstraintsAsPenalty
from pymoo.optimize import minimize
from pymoo.problems.single import G1

problem = G1()

problem = ConstraintsAsPenalty(problem, penalty=100.0)

algorithm = DE()

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)

opt = res.opt

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (opt.get('X'), opt.get('__F__'), opt.get('__CV__')))
