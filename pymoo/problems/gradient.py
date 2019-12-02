import autograd.numpy as anp
from autograd.core import VJPNode, vspace, backward_pass
from autograd.tracer import new_box, isbox, toposort


# runs the function by making sure the calculations are traced using autograd
def run_and_trace(fun, x, *args, **kwargs):
    start_node = VJPNode.new_root()

    start_box = new_box(x, 0, start_node)
    out = fun(start_box, *args, **kwargs)

    return start_box, out



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

