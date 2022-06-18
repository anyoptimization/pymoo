from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.termination.ftol import SingleObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination

problem = get_problem("rastrigin")
algorithm = GA(pop_size=100)
termination = RobustTermination(SingleObjectiveSpaceTermination())

res = minimize(problem,
               algorithm,
               termination,
               pf=problem.pareto_front(),
               seed=1,
               verbose=True)

print(res.opt.get("F"))
print(res.algorithm.n_gen)
