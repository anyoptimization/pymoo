from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter

problem = get_problem("dtlz4")

algorithm = AGEMOEA2()

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=3,
               verbose=True)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()

