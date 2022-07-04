from pymoo.algorithms.moo.rvea import RVEA
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

problem = get_problem("dtlz1", n_obj=3)

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

algorithm = RVEA(ref_dirs)

res = minimize(problem,
               algorithm,
               termination=('n_gen', 400),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(problem.pareto_front(ref_dirs), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()