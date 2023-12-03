from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.core.variable import Float
from pymoo.problems.customsphere import CustomSphere
import numpy as np

problem = CustomSphere(vtype=Float(size=10, bounds=(-1, 1)))


def test_nelder_sphere():
    result = NelderMead().setup(problem).run()
    assert np.allclose(result.f, 0.0, atol=1e-6)


def test_nelder_sphere_init_sol():
    init_sol = problem.vtype.random()
    result = NelderMead().setup(problem, init_sol=init_sol).run()
    assert np.allclose(result.f, 0.0, atol=1e-6)


def test_nelder_sphere_init_sol_as_vector():
    init_sol = problem.vtype.random().get()
    result = NelderMead().setup(problem, init_sol=init_sol).run()
    assert np.allclose(result.f, 0.0, atol=1e-6)
