from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.univariate.golden import GoldenSectionSearch
from pymoo.core.variable import Float
from pymoo.problems.customsphere import CustomSphere
import numpy as np

problem = CustomSphere(vtype=Float(size=10, bounds=(-1, 1)))


def test_golden_sphere():
    result = GoldenSectionSearch(n_max_iter=100).setup(problem).run()
    assert np.allclose(result.f, 0.0)
