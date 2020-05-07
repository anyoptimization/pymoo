from pymoo.algorithms.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("DASCMOP1", 12)

ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_points=91)
algorithm = NSGA3(ref_dirs)

res = minimize(problem,
               algorithm,
               ("n_gen", 600),
               verbose=True
               )

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()
