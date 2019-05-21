from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import nsga2
from pymoo.util import plotting

# create the algorithm object
method = nsga2(pop_size=100, elimate_duplicates=True)

# execute the optimization
res = minimize(get_problem("zdt1"),
               method,
               ('n_gen', 200),
               seed=1,
               verbose=False)

plotting.plot(res.F, no_fill=True)
