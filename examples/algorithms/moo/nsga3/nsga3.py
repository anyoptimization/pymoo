from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

problem = get_problem("dtlz1")

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

algorithm = NSGA3(pop_size=92,
                  ref_dirs=ref_dirs)

res = minimize(problem,
               algorithm,
               seed=1,
               termination=('n_gen', 600),
               verbose=True)

Scatter().add(res.F).show()
