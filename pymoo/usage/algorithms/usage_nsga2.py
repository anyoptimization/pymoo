from pymoo.algorithms.nsga2 import nsga2
from pymoo.analytics.visualization.scatter import Scatter
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("zdt3")

method = nsga2(pop_size=100, elimate_duplicates=True)

# execute the optimization
res = minimize(problem,
               method,
               ('n_gen', 200),
               seed=1,
               verbose=False)

scatter = Scatter()
scatter.add(problem.pareto_front(n_points=200, flatten=False), plot_type="line", color="black", alpha=0.7)
scatter.add(res.F, color="red")
scatter.show()
