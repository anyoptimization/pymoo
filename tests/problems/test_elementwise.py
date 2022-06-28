import numpy as np

from pymoo.core.problem import Problem, ElementwiseProblem


class MyElementwiseProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = (2 * x).sum()
        out["anyvar"] = 5.0


class MyProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum(2 * x, axis=1)
        out["anyvar"] = [5.0] * len(x)


def test_elementwise_evaluation():
    X = np.random.random((100, 2))

    vectorized = MyProblem()
    elementwise = MyElementwiseProblem()
    np.testing.assert_allclose(vectorized.evaluate(X), elementwise.evaluate(X))


def test_misc_value():
    X = np.random.random((100, 2))
    vectorized = MyProblem()
    elementwise = MyElementwiseProblem()

    a = vectorized.evaluate(X, return_values_of=["anyvar"])
    b = elementwise.evaluate(X, return_values_of=["anyvar"])
    np.testing.assert_allclose(a, b)
