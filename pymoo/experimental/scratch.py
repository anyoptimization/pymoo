
from unittest.mock import MagicMock

from pymoo.algorithms.nsga2 import NSGA2

from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util.optimum import filter_optimum

problem = get_problem("zdt1")

algorithm = NSGA2(pop_size=100)
algorithm._set_optimum = MagicMock(return_value=None)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

opt = filter_optimum(res.algorithm.pop)

print(opt)


