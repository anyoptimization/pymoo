# START nsga3
from pymoo.algorithms.nsga3 import nsga3
from pymoo.analytics.visualization.scatter import Scatter
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util.reference_direction import UniformReferenceDirectionFactory

# create the reference directions to be used for the optimization
ref_dirs = UniformReferenceDirectionFactory(3, n_points=91).do()

# create the algorithm object
method = nsga3(pop_size=92,
               ref_dirs=ref_dirs)

# execute the optimization
res = minimize(get_problem("dtlz1"),
               method,
               seed=1,
               termination=('n_gen', 600))

scatter = Scatter(angle=(45, 45)).add(res.F).do().show()
# END nsga3

# START inverted_dtzl_1
res = minimize(get_problem("dtlz1_-1"),
               method,
               seed=1,
               termination=('n_gen', 600))

Scatter(angle=(45, 45)).add(res.F).show()
# END inverted_dtzl_1
