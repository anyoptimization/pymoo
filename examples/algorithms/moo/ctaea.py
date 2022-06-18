from pymoo.algorithms.moo.ctaea import CTAEA

from pymoo.optimize import minimize
from pymoo.problems.multi import Carside
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

problem = Carside()

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

algorithm = CTAEA(ref_dirs=ref_dirs)

res = minimize(problem,
               algorithm,
               ('n_gen', 600),
               seed=1,
               verbose=True
               )

Scatter().add(res.F, facecolor="none", edgecolor="red").show()
