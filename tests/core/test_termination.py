import numpy as np

from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.core.termination import MaximumIterationTermination
from pymoo.core.variable import Float
from pymoo.problems.customsphere import CustomSphere

problem = CustomSphere(vtype=Float(size=10, bounds=(-1, 1)))


def test_termination_iter():
    termination = MaximumIterationTermination(max_iter=1000)
    result = NelderMead(termination=termination).setup(problem).run()
    assert np.allclose(result.f, 0.0)
