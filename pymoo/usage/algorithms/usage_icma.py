from pymoo.algorithms.moo.gde3 import GDE3
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt2")

algorithm = GDE3(pop_size=100)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()
