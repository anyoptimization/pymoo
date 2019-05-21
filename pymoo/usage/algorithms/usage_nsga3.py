# START nsga3
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util import plotting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.algorithms.nsga3 import nsga3
import matplotlib.pyplot as plt

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


ax = plotting.plot(res.F, show=False, alpha=1.0)
ax.view_init(45, 45)
plt.show()
# END nsga3

# START inverted_dtzl_1
res = minimize(get_problem("dtlz1_-1"),
               method,
               seed=1,
               termination=('n_gen', 600))

ax = plotting.plot(res.F, show=False, alpha=1.0)
ax.view_init(25, 45)
plt.show()
# END inverted_dtzl_1