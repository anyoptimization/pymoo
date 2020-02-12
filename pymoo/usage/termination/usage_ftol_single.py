from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_problem, get_termination
from pymoo.optimize import minimize

problem = get_problem("rastrigin")
algorithm = GA(pop_size=100)
termination = get_termination("f_tol_s")

res = minimize(problem,
               algorithm,
               termination,
               pf=problem.pareto_front(),
               seed=1,
               verbose=True)

print(res.opt.get("F"))
print(res.algorithm.n_gen)
