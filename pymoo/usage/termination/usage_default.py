from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem, get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt5")
algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               get_termination("default"),
               pf=False,
               seed=2,
               verbose=True)

print(res.algorithm.n_gen)
plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black")
plot.add(res.F, color="red", alpha=0.8, s=20)
plot.show()



