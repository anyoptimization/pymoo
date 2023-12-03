from copy import deepcopy

import numpy as np

from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.core.termination import MaximumIterationTermination
from pymoo.core.variable import Float
from pymoo.problems.customsphere import CustomSphere

problem = CustomSphere(vtype=Float(size=10, bounds=(-1, 1)), constrained=True)


# termination = MaximumIterationTermination(200)
# algorithm = G3PCX(termination=termination)

algorithm = NelderMead()

result = deepcopy(algorithm).setup(problem, verbose=True).run()

print(result.f)

assert np.allclose(result.f, 0.0, atol=1e-6)
