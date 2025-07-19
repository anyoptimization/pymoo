import numpy as np

from pymoo.core.problem import ElementwiseEvaluationFunction, MetaProblem
from pymoo.gradient import activate, deactivate


class ElementwiseEvaluationFunctionWithGradient(ElementwiseEvaluationFunction):

    def __init__(self, problem, backend='autograd', args=(), kwargs={}):
        super().__init__(problem, args, kwargs)
        self.backend = backend

    def __call__(self, x):
        f = super().__call__

        activate(self.backend)
        if self.backend == "jax":
            from pymoo.gradient.grad_jax import jax_elementwise_value_and_grad
            out, grad = jax_elementwise_value_and_grad(f, x)
        elif self.backend == "autograd":
            from pymoo.gradient.grad_autograd import autograd_elementwise_value_and_grad
            out, grad = autograd_elementwise_value_and_grad(f, x)
        else:
            raise Exception("Unknown backend %s" % self.backend)
        deactivate()

        for k, v in grad.items():
            out["d" + k] = np.array(v)

        return out


class ElementwiseAutomaticDifferentiation(MetaProblem):

    def __init__(self, problem, backend='autograd', copy=True):
        if not problem.elementwise:
            raise Exception("Elementwise automatic differentiation can only be applied to elementwise problems.")

        super().__init__(problem, copy)
        self.backend = backend

        # Set the elementwise_func to the class itself - it will handle the signature
        self.elementwise_func = self._create_elementwise_func

    def _create_elementwise_func(self, problem, args, kwargs):
        """Create an elementwise function that matches the expected signature"""
        return ElementwiseEvaluationFunctionWithGradient(self.__object__, self.backend, args, kwargs)


class AutomaticDifferentiation(MetaProblem):

    def __init__(self, problem, backend='autograd', **kwargs):
        super().__init__(problem, **kwargs)
        self.backend = backend

    def do(self, x, return_values_of, *args, **kwargs):

        vals_not_grad = [v for v in return_values_of if not v.startswith("d")]

        class F:

            def __init__(self, wrapped_problem):
                self.__object__ = wrapped_problem

            def __call__(self, xp):
                return self.__object__.do(xp, vals_not_grad, *args, **kwargs)

        f = F(self.__wrapped__)

        activate(self.backend)
        if self.backend == "jax":
            from pymoo.gradient.grad_jax import jax_vectorized_value_and_grad
            out, grad = jax_vectorized_value_and_grad(f, x)
        elif self.backend == "autograd":
            from pymoo.gradient.grad_autograd import autograd_vectorized_value_and_grad
            out, grad = autograd_vectorized_value_and_grad(f, x)
        else:
            raise Exception("Unknown backend %s" % self.backend)
        deactivate()

        for k, v in grad.items():
            out["d" + k] = v

        return out
