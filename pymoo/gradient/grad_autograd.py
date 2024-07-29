import warnings
import numpy as np

try:
    import autograd.numpy as anp
    from autograd.core import VJPNode, backward_pass
    from autograd.tracer import new_box, isbox
except:
    print("autograd only supports numpy < 2.0.0 versions.")


def value_and_grad(*args, **kwargs):
    from autograd import value_and_grad as vag
    return vag(*args, **kwargs)


def log(*args, **kwargs):
    return anp.log(*args, **kwargs)


def sqrt(*args, **kwargs):
    return anp.sqrt(*args, **kwargs)


def row_stack(*args, **kwargs):
    return anp.row_stack(*args, **kwargs)


def triu_indices(*args, **kwargs):
    return anp.triu_indices(*args, **kwargs)


def run_and_trace(f, x):
    start_node = VJPNode.new_root()

    start_box = new_box(x, 0, start_node)
    out = f(start_box)

    return out, start_box


def autograd_elementwise_value_and_grad(f, x):
    out, pullback = run_and_trace(f, x)

    jac = dict()
    for name in out:
        val = out[name]

        if val is not None:

            if len(val.shape) == 0:
                val = anp.array([val])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # the backward pass is done for each objective function once
                grad = []
                for j in range(len(val)):
                    b = np.zeros(val.shape)
                    b[j] = 1

                    n = new_box(b, 0, VJPNode.new_root())
                    _grad = backward_pass(n, val._node)
                    grad.append(_grad)

                out[name] = np.array(val._value)
                jac[name] = anp.stack(grad, axis=0)._value

    return out, jac


def autograd_vectorized_value_and_grad(f, x):
    end, start = run_and_trace(f, x)

    out, jac = dict(), dict()

    end = {k: v for k, v in end.items() if v is not None}

    for name, val in end.items():

        v = val
        if hasattr(v, "_value"):
            v = np.array(v._value)
        out[name] = v

        # if the end_box is not a box - autograd can not track back
        if not isbox(val):
            n, m = val.shape
            jac[name] = np.zeros((n, m, x.shape[1]))

        else:

            # the backward pass is done for each objective function once
            grad = []
            for j in range(val.shape[1]):
                b = anp.zeros(val.shape)
                b[:, j] = 1
                n = new_box(b, 0, VJPNode.new_root())
                _grad = backward_pass(n, val._node)
                grad.append(_grad)

            jac[name] = anp.stack(grad, axis=1)._value

    return out, jac
