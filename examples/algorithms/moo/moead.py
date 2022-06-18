from pymoo.algorithms.moo.moead import MOEAD, ParallelMOEAD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

problem = get_problem("dtlz2")

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

algorithm = MOEAD(
    ref_dirs,
    n_neighbors=15,
    prob_neighbor_mating=0.7,
    seed=1,
    verbose=False
)

res = minimize(problem, algorithm, termination=('n_gen', 200), verbose=True)

Scatter().add(res.F).show()
