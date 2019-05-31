from pymoo.algorithms.nsga2 import nsga2
from pymoo.factory import get_problem, get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import scatter

problem = get_problem("zdt3")
algorithm = nsga2(pop_size=100)
termination = get_termination("ftol", tol=0.001, n_last=20, n_max_gen=1000, nth_gen=10)

res = minimize(problem,
               algorithm,
               termination=termination,
               pf=problem.pareto_front(),
               seed=1,
               verbose=False)

print(algorithm.n_gen)
plot = scatter(title="ZDT3")
plot.add(problem.pareto_front(use_cache=False, flatten=False), plot_type="line", color="black")
plot.add(res.F, color="red", alpha=0.8, s=20)
plot.show()
