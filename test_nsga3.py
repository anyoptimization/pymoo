from pymoo.algorithms.adapted_nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import time
# create the reference directions to be used for the optimization
ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=12)

# create the algorithm object
algorithm = NSGA3(pop_size=92,
                  ref_dirs=ref_dirs)
start = time.time()
# execute the optimization
res = minimize(get_problem("dtlz2", n_obj=5),
               algorithm,
               seed=1,
               termination=('n_gen', 600))
end = time.time()

print('Elapsed time {}'.format(end - start))
