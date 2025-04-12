import numpy as np

from pymoo.core.meta import Meta
from pymoo.core.problem import Problem


def calc_complex_gradient(problem, return_values_of, x, eps, *args, **kwargs):
    xp = x + np.eye(len(x)) * complex(0, eps)
    out = problem.do(xp, return_values_of, *args, **kwargs)

    grad = {}
    for name, value in out.items():
        grad[name] = np.imag(value / eps).T

    return grad


class ComplexNumberGradient(Meta, Problem):

    def __init__(self, object, eps=1e-8, **kwargs):
        super().__init__(object, **kwargs)
        self.eps = eps

    def do(self, X, return_values_of, *args, **kwargs):
        out = self.__object__.do(X, return_values_of, *args, **kwargs)

        vals_not_grad = [v for v in return_values_of if not v.startswith("d")]

        for i, x in enumerate(X):
            grad = calc_complex_gradient(self.__object__, vals_not_grad, x, self.eps, *args, **kwargs)

            for name, value in grad.items():
                out['d' + name][i] = value

        return out
