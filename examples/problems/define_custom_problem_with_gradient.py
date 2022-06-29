import numpy as np

import numpy as np

from pymoo.core.problem import Problem

"""
============================================
Traditional Definition
============================================

In case the gradient is provided directly by the evaluation function this is possible.
The evaluation_of variable defines what variables are calculated by the evaluation function.

"""


class GradientDirectlyImplemented(Problem):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=0, xl=0, xu=1, vtype=float, evaluation_of=["F", "dF"],
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:, 1:], axis=1)
        f2 = g * (1 - np.power((f1 / g), 0.5))

        out["F"] = np.column_stack([f1, f2])

        if "dF" in out:
            dF = np.zeros([x.shape[0], self.n_obj, self.n_var], dtype=float)
            dF[:, 0, 0], dF[:, 0, 1:] = 1, 0
            dF[:, 1, 0] = -0.5 * np.sqrt(g / x[:, 0])
            dF[:, 1, 1:] = ((9 / (self.n_var - 1)) * (1 - 0.5 * np.sqrt(x[:, 0] / g)))[:, None]
            out["dF"] = dF


problem = GradientDirectlyImplemented(n_var=10)
F, dF = problem.evaluate(np.random.random((100, 10)), return_values_of=["F", "dF"])


"""
============================================
Automatic Differentiation
============================================

Autograd can be used to calculate the gradient. Therefore, you have to use the correct import.
Please note to distinguish between numpy and autograd.numpy it is good practice to import it as np.

If the function evaluations asks for the gradient autograd does its job and ONLY the function evaluation needs
to be added to the problem.

"""


class AutomaticDifferentiation(Problem):

    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=0, xl=0, xu=1, vtype=float, evaluation_of=["F"],
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:, 1:], axis=1)
        f2 = g * (1 - np.power((f1 / g), 0.5))

        out["F"] = np.column_stack([f1, f2])


problem = AutomaticDifferentiation(n_var=10)
F, dF = problem.evaluate(np.random.random((100, 10)), return_values_of=["F", "dF"])
