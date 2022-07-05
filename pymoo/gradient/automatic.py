import numpy as np

from pymoo.core.meta import Meta
from pymoo.core.problem import Problem, ElementwiseEvaluationFunction


class ElementwiseEvaluationFunctionWithGradient(ElementwiseEvaluationFunction):

    def __call__(self, x):
        f = super().__call__

        from pymoo.gradient import TOOLBOX

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

    def __init__(self, problem, copy=True):
        if not problem.elementwise:
            raise Exception("Elementwise automatic differentiation can only be applied to elementwise problems.")

        super().__init__(problem, copy)
        self.elementwise_func = ElementwiseEvaluationFunctionWithGradient


class AutomaticDifferentiation(Meta, Problem):

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
