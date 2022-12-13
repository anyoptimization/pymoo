from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Integer, Binary


def test_mixed_vars_bounds():
    class MyProblem(Problem):
        def __init__(self, **kwargs):
            variables = {'x': Integer(bounds=(1, 10)),
                         'y': Real(bounds=(0.1, 0.99)),
                         'z': Binary(bounds=(0, 1))
                         }
            super().__init__(vars=variables,
                             n_obj=3,
                             **kwargs)

    problem = MyProblem()

    assert problem.has_bounds()
    assert problem.xl == dict(x=1, y=0.1, z=0)
    assert problem.xu == dict(x=10, y=0.99, z=1)

