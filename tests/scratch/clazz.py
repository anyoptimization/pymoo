import numpy as np

from pymoo.algorithms.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.pcp import PCP

p = "dtlz5"

# create the reference directions to be used for the optimization
ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=6)

problem = get_problem(p, n_obj=5)

# create the algorithm object
algorithm = NSGA3(ref_dirs=ref_dirs)

# execute the optimization
ret = minimize(problem,
               algorithm,
               # pf=problem.pareto_front(ref_dirs),
               seed=1,
               termination=('n_gen', 1000),
               verbose=True)

np.savetxt("%s_%s.f" % (p, len(ref_dirs)), ret.F)

PCP().add(ret.F).show()