import numpy as np

from pymoo.core.meta import Meta
from pymoo.core.problem import Problem, ElementwiseProblem


class ElementwiseEvaluationWithGradient:

    def __init__(self, func) -> None:
        super().__init__()
        self.func = func

    def __call__(self, problem, x, args, kwargs):
        from pymoo.gradient import TOOLBOX

        f = lambda xp: self.func(problem, xp, args, kwargs)

        if TOOLBOX == "jax.numpy":
            from pymoo.gradient.grad_jax import jax_elementwise_value_and_grad
            out, grad = jax_elementwise_value_and_grad(f, x)

        elif TOOLBOX == "autograd.numpy":
            from pymoo.gradient.grad_autograd import autograd_elementwise_value_and_grad
            out, grad = autograd_elementwise_value_and_grad(f, x)

        for k, v in grad.items():
            out["d" + k] = np.array(v)

        return out


class ElementwiseAutomaticDifferentiation(Meta, Problem):

    def __init__(self, object, copy=True):
        super().__init__(object, copy)
        self.func_elementwise_eval = ElementwiseEvaluationWithGradient(self.func_elementwise_eval)

    def do(self, X, return_values_of, *args, **kwargs):
        ret = super().do(X, return_values_of, *args, **kwargs)
        return ret


class VectorizedAutomaticDifferentiation(Meta, Problem):

    def do(self, x, return_values_of, *args, **kwargs):
        from pymoo.gradient import TOOLBOX

        vals_not_grad = [v for v in return_values_of if not v.startswith("d")]
        f = lambda xp: self.__object__.do(xp, vals_not_grad, *args, **kwargs)

        if TOOLBOX == "jax.numpy":
            from pymoo.gradient.grad_jax import jax_vectorized_value_and_grad
            out, grad = jax_vectorized_value_and_grad(f, x)

        elif TOOLBOX == "autograd.numpy":
            from pymoo.gradient.grad_autograd import autograd_vectorized_value_and_grad
            out, grad = autograd_vectorized_value_and_grad(f, x)

        for k, v in grad.items():
            out["d" + k] = v

        return out


class AutomaticDifferentiation(Meta, Problem):

    def __new__(cls, object, **kwargs):

        if isinstance(object, ElementwiseProblem):
            return ElementwiseAutomaticDifferentiation(object)
        elif isinstance(object, Problem):
            return VectorizedAutomaticDifferentiation(object)
        else:
            raise Exception("For AutomaticDifferentiation the problem must be either Problem or ElementwiseProblem.")
