# START ctaea
from pymoo.algorithms.ctaea import CTAEA
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("DASCMOP1", 2)

# create the reference directions to be used for the optimization
ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_points=91)

# create the algorithm object
algorithm = CTAEA(ref_dirs=ref_dirs)

# execute the optimization
res = minimize(problem,
               algorithm,
               ('n_gen', 600),
               seed=1,
               verbose=True
               )

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()
# END ctaea

# START carside
problem = get_problem("carside")
ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_points=91)
algorithm = CTAEA(ref_dirs=ref_dirs)

res = minimize(problem,
               algorithm,
               ('n_gen', 600),
               seed=1,
               verbose=True
               )

Scatter().add(res.F).show()
# END carside
