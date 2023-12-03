from pymoo.algorithms.random_search import RandomSearch
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.core.problem import Problem, ProblemType
from pymoo.core.variable import Float, Binary, Mixed
from pymoo.problems.customsphere import CustomSphere
import numpy as np

problem = CustomSphere(vtype=Float(size=2, bounds=(-1, 1)))


def test_random_mixed():
    class MyProblem(Problem):

        def __init__(self) -> None:
            vars = (Float(size=1, bounds=(0, 1)), Binary())
            vtype = Mixed(vars)
            super().__init__(vtype, parallel=False)

        def _evaluate(self, x: np.ndarray, out: dict) -> None:
            floats, bin = x

            if not bin:
                f = floats.sum()
            else:
                f = 100 + floats.sum()

            out["F"][:] = f ** 2

    problem = MyProblem()
    result = RandomSearch(n_max_iter=100).setup(problem).run()
    assert np.allclose(result.f, 0.0, atol=1e-5)


def test_random_sphere_init_sol():
    algorithm = RandomSearch(n_max_iter=1000).setup(problem)
    result = algorithm.run()
    print(result.x)
    assert np.allclose(result.f, 0.0, atol=1e-5)
