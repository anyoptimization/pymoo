# START ctaea
from pymoo.algorithms.ctaea import CTAEA
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# create the reference directions to be used for the optimization
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

# create the algorithm object
algorithm = CTAEA(ref_dirs=ref_dirs)

# execute the optimization
res = minimize(get_problem("c3dtlz4"),
               algorithm,
               seed=1,
               termination=('n_gen', 600))

Scatter().add(res.F).show()
# END ctaea

# START carside
res = minimize(get_problem("carside"),
               algorithm,
               seed=1,
               termination=('n_gen', 600))

Scatter().add(res.F).show()
# END carside
