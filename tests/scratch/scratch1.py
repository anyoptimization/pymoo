from pymoo.algorithms.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.util.normalization import normalize
from pymoo.visualization.scatter import Scatter

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

problem = get_problem("dtlz1")
pf = problem.pareto_front(ref_dirs)

algorithm = NSGA3(ref_dirs)

res = minimize(problem,
               algorithm,
               ('n_gen', 400),
               seed=1,
               verbose=True)

plot = Scatter()
plot.add(pf, plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()


