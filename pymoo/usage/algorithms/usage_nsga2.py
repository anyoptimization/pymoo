from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt3")
pf = problem.pareto_front(n_points=200, flatten=False, use_cache=False)

algorithm = NSGA2(pop_size=100, elimate_duplicates=True)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(pf, plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()
