from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# create the algorithm object
algorithm = AGEMOEA2(pop_size=92)

# execute the optimization
res = minimize(get_problem("dtlz2"),
               algorithm,
               termination=('n_gen', 600),
               verbose=True,
               seed=1,
               )

Scatter().add(res.F).show()
