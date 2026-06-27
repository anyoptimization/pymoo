"""Autograd-based automatic differentiation backend."""

import warnings

import numpy as np

try:
    import autograd.numpy as anp
    from autograd.core import VJPNode, backward_pass
    from autograd.tracer import new_box, isbox
except Exception:  # noqa: E722
    print("autograd only supports numpy < 2.0.0 versions.")


def value_and_grad(*args, **kwargs):  # type: ignore[name-defined]
    """Compute value and gradient using autograd.

    Args:
        *args: Arguments to autograd.value_and_grad.
        **kwargs: Keyword arguments to autograd.value_and_grad.

    Returns:
        Value and gradient computation function.
    """
    from autograd import value_and_grad as vag

    return vag(*args, **kwargs)


def log(*args, **kwargs):  # type: ignore[name-defined]
    """Natural logarithm."""
    return anp.log(*args, **kwargs)


def sqrt(*args, **kwargs):  # type: ignore[name-defined]
    """Square root."""
    return anp.sqrt(*args, **kwargs)


def row_stack(*args, **kwargs):  # type: ignore[name-defined]
    """Stack arrays vertically."""
    return anp.row_stack(*args, **kwargs)


def triu_indices(*args, **kwargs):  # type: ignore[name-defined]
    """Return indices of upper triangular matrix."""
    return anp.triu_indices(*args, **kwargs)


def run_and_trace(f, x):  # type: ignore[name-defined]
    """Run function and trace computation graph.

    Args:
        f: Function to trace.
        x: Input value.

    Returns:
        Output value and traced input box.
    """
    start_node = VJPNode.new_root()

    start_box = new_box(x, 0, start_node)
    out = f(start_box)

    return out, start_box


def autograd_elementwise_value_and_grad(f, x):  # type: ignore[name-defined]
    """Compute value and gradient for elementwise function.

    Args:
        f: Function to differentiate.
        x: Input value.

    Returns:
        Dictionary with output and jacobian dictionary.
    """
    out, pullback = run_and_trace(f, x)

    jac = dict()
    for name in out:
        val = out[name]

        if val is not None:
            if len(val.shape) == 0:
                val = anp.array([val])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

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


def autograd_vectorized_value_and_grad(f, x):  # type: ignore[name-defined]
    """Compute value and gradient for vectorized function.

    Args:
        f: Function to differentiate.
        x: Input array of shape (n, p).

    Returns:
        Dictionary with output and jacobian dictionary.
    """
    end, start = run_and_trace(f, x)

    out, jac = dict(), dict()

    end = {k: v for k, v in end.items() if v is not None}

    for name, val in end.items():
        v = val
        if hasattr(v, "_value"):
            v = np.array(v._value)
        out[name] = v

        if not isbox(val):
            n, m = val.shape
            jac[name] = np.zeros((n, m, x.shape[1]))

        else:
            grad = []
            for j in range(val.shape[1]):
                b = anp.zeros(val.shape)
                b[:, j] = 1
                n = new_box(b, 0, VJPNode.new_root())
                _grad = backward_pass(n, val._node)
                grad.append(_grad)

            jac[name] = anp.stack(grad, axis=1)._value

    return out, jac
