from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere
from pymoo.termination.fmin import MinimumFunctionValueTermination

problem = Sphere()

algorithm = PSO()

termination = MinimumFunctionValueTermination(1e-5)

res = minimize(problem,
               algorithm,
               termination,
               pf=problem.pareto_front(),
               seed=1,
               verbose=True)
