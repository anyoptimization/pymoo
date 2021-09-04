import warnings

import autograd
import autograd.numpy as anp
from autograd.core import VJPNode, vspace, backward_pass
from autograd.tracer import new_box, isbox

from pymoo.core.problem import ElementwiseProblem, out_to_ndarray, check
from pymoo.problems.meta import MetaProblem


def out_to_numpy(out):
    for key in out.keys():
        if type(out[key]) == autograd.numpy.numpy_boxes.ArrayBox:
            out[key] = out[key]._value


def get_deriv(out):
    deriv = []
    if "dF" in out:
        deriv.append("F")
    if "dG" in out:
        deriv.append("G")
    return deriv


def run_and_trace(fun, x, *args, **kwargs):
    start_node = VJPNode.new_root()

    start_box = new_box(x, 0, start_node)
    out = fun(start_box, *args, **kwargs)

    return start_box, out


def calc_jacobian_elem(start, end):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = anp.ones(start.shape)
        n = new_box(b, 0, VJPNode.new_root())
        jac = backward_pass(n, end._node)
        return jac._value


def calc_jacobian(start, end):
    # if the end_box is not a box - autograd can not track back
    if not isbox(end):
        return vspace(start.shape).zeros()

    # the final jacobian matrices
    jac = []

    # the backward pass is done for each objective function once
    for j in range(end.shape[1]):
        b = anp.zeros(end.shape)
        b[:, j] = 1
        n = new_box(b, 0, VJPNode.new_root())
        _jac = backward_pass(n, end._node)
        jac.append(_jac)

    jac = anp.stack(jac, axis=1)

    return jac


def autograd_elementwise_eval(problem, X, out, *args, **kwargs):
    deriv = get_deriv(out)

    if len(deriv) > 0:
        out["__autograd__"], _ = run_and_trace(problem._evaluate, X, out, *args, **kwargs)

        for key in deriv:
            val = out.get(key)
            if val is not None:

                if not isinstance(val, anp.ndarray):
                    val = anp.array([val])

                out["d" + key] = calc_jacobian_elem(out["__autograd__"], val)[None, :]

        # make sure all results are pure numpy arrays
        out_to_numpy(out)

    out_to_ndarray(out)
    check(problem, X, out)

    return out


class AutomaticDifferentiation(MetaProblem):

    def __init__(self, problem):
        super().__init__(problem)

    def do(self, X, out, *args, **kwargs):

        problem = self.problem

        if isinstance(problem, ElementwiseProblem):

            # store the original function for evaluation
            func = problem.func_elementwise_eval

            # set a new elementwise function, do the evaluation and revert to default
            problem.func_elementwise_eval = autograd_elementwise_eval
            problem.do(X, out, *args, **kwargs)

            # revert to the old function evaluation
            problem.func_elementwise_eval = func

        else:
            deriv = get_deriv(out)

            if len(deriv) > 0:
                out["__autograd__"], _ = run_and_trace(self.problem.do, X, out, *args, **kwargs)

                for key in deriv:
                    val = out.get(key)

                    if val is not None:

                        if val.ndim == 1:
                            val = val[:, None]

                        jac = calc_jacobian(out["__autograd__"], val)
                        out["d" + key] = jac

                # make sure all results are pure numpy arrays
                out_to_numpy(out)

            else:
                self.problem.do(X, out, *args, **kwargs)
