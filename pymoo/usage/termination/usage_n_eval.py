from pymoo.algorithms.nsga2 import nsga2
from pymoo.factory import get_problem, get_termination
from pymoo.optimize import minimize

problem = get_problem("zdt3")
termination = get_termination("n_eval", 300)

res = minimize(problem,
               nsga2(pop_size=100),
               termination=termination,
               pf=problem.pareto_front(),
               seed=1,
               verbose=True)