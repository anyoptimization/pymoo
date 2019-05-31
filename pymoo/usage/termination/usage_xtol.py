from pymoo.algorithms.nsga2 import nsga2
from pymoo.factory import get_problem, get_termination
from pymoo.optimize import minimize

problem = get_problem("zdt3")
algorithm = nsga2(pop_size=100)
termination = get_termination("xtol", tol=0.001, n_last=20, n_max_gen=None, nth_gen=10)

res = minimize(problem,
               algorithm,
               termination=termination,
               pf=problem.pareto_front(),
               seed=1,
               verbose=False)

print(algorithm.n_gen)