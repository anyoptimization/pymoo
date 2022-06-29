import sys

import numpy as np
from matplotlib import patches
from matplotlib.collections import PatchCollection


def get_circle_points(n_points):
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)])


def default_number_to_text(val):
    if val > 1e3:
        return "{:.2e}".format(val)
    else:
        return "{:.2f}".format(val)


def in_notebook():
    return 'ipykernel' in sys.modules


def get_uniform_points_around_circle(n):
    t = 2 * np.pi * np.arange(n) / n
    s = np.column_stack([np.cos(t), np.sin(t)])
    return s


def plot_circle(ax, center=0, radius=1, **kwargs):
    P = get_circle_points(5000)
    P = (P + center) * radius
    ax.plot(P[:, 0], P[:, 1], **kwargs)


def plot_radar_line(ax, x, **kwargs):
    x = np.row_stack([x, x[0]])
    ax.plot(x[:, 0], x[:, 1], **kwargs)


def plot_axes_arrow(ax, X, extend_factor=1.0, **kwargs):
    for (x, y) in X:
        ax.arrow(0, 0, x * extend_factor, y * extend_factor, **kwargs)


def plot_axes_lines(ax, X, extend_factor=1.0, **kwargs):
    for (x, y) in X:
        ax.plot([0, x * extend_factor], [0, y * extend_factor], **kwargs)


def plot_polygon(ax, x, **kwargs):
    ax.add_collection(PatchCollection([patches.Polygon(x, True)], **kwargs))


def plot_axis_labels(ax, endpoints, labels, margin=0.035, size='small', **kwargs):
    for k in range(len(labels)):
        xy = endpoints[k]

        if xy[0] < 0.0:
            x = xy[0] - margin
            ha = "right"
        else:
            x = xy[0] + margin
            ha = "left"

        if xy[1] < 0.0:
            y = xy[1] - margin
            va = "top"
        else:
            y = xy[1] + margin
            va = "bottom"

        ax.text(x, y, labels[k], ha=ha, va=va, size=size, **kwargs)


def equal_axis(ax):
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.axis('equal')


def no_ticks(ax):
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_frame_on(False)


def normalize(data, bounds, reverse=False, return_bounds=False):
    from pymoo.util.normalization import normalize as _normalize

    _F = np.row_stack([e[0] for e in data])
    if bounds is None:
        bounds = (_F.min(axis=0), _F.max(axis=0))

    to_plot = []
    for k in range(len(data)):
        F = _normalize(data[k][0], bounds[0], bounds[1])

        if reverse:
            F = 1 - F

        to_plot.append([F, data[k][1]])

    if return_bounds:
        return to_plot, bounds
    else:
        return to_plot


def parse_bounds(bounds, n_dim):
    if bounds is not None:
        bounds = np.array(bounds, dtype=float)
        if bounds.ndim == 1:
            bounds = bounds[None, :].repeat(n_dim, axis=0).T
    return bounds


def radviz_pandas(F):
    import pandas as pd
    df = pd.DataFrame([x for x in F], columns=["X%s" % k for k in range(F.shape[1])])
    df["class"] = "Points"
    return pd.plotting.radviz(df, "class")

