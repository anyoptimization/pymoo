from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems.multi.srn import SRN
from pymoo.visualization.scatter import Scatter

problem = SRN()


algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               # ('n_gen', 1000),
               seed=1,
               verbose=True)

plot = Scatter()
plot.add(problem.pareto_set(), plot_type="line", color="black", alpha=0.7)
plot.add(res.X, color="red")
plot.show()


plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()
