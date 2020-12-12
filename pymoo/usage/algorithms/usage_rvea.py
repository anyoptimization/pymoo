from pymoo.algorithms.moo.rvea import RVEA
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("dtlz1", n_obj=3)

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

algorithm = RVEA(ref_dirs)

pf = problem.pareto_front(ref_dirs)

res = minimize(problem,
               algorithm,
               ('n_gen', 250),
               seed=1,
               pf=pf,
               verbose=True)

plot = Scatter()
plot.add(pf, plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()



