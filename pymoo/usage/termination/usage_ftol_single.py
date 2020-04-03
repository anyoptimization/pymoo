from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util.termination.f_tol_single import SingleObjectiveSpaceToleranceTermination

problem = get_problem("rastrigin")
algorithm = GA(pop_size=100)
termination = SingleObjectiveSpaceToleranceTermination()

res = minimize(problem,
               algorithm,
               termination,
               pf=problem.pareto_front(),
               seed=1,
               verbose=True)

print(res.opt.get("F"))
print(res.algorithm.n_gen)
