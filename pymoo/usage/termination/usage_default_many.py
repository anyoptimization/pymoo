from pymoo.algorithms.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions, get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("dtlz3", None, 3, k=10)

# create the reference directions to be used for the optimization
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

# create the algorithm object
algorithm = NSGA3(pop_size=92, ref_dirs=ref_dirs)

res = minimize(problem,
               algorithm,
               get_termination("default", n_last=25),
               pf=True,
               seed=2,
               verbose=True)

print(res.algorithm.n_gen)
plot = Scatter(title="DTLZ3")
plot.add(res.F, color="red", alpha=0.8, s=20)
plot.show()

