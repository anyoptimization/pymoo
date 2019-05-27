from pymop.factory import get_problem

from pymoo.algorithms.nsga3 import nsga3
from pymoo.optimize import minimize
from pymoo.util.reference_direction import MultiLayerReferenceDirectionFactory, UniformReferenceDirectionFactory

pop_size = 276
ref_dirs = MultiLayerReferenceDirectionFactory([
    UniformReferenceDirectionFactory(10, n_partitions=3, scaling=1.0),
    UniformReferenceDirectionFactory(10, n_partitions=2, scaling=0.5)]).do()

problem = get_problem("dtlz3", n_obj=10)
pf = problem.pareto_front(ref_dirs)

# create the algorithm object
method = nsga3(ref_dirs, pop_size=pop_size)

# execute the optimization
res = minimize(problem,
               method,
               seed=1,
               pf=pf,
               termination=('n_gen', 400),
               verbose=True)
