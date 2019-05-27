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


def plot_axis_labels(ax, endpoints, labels, margin=0.035):
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

        ax.text(x, y, labels[k], ha=ha, va=va, size='small')


def radviz_pandas(F):
    import pandas as pd
    df = pd.DataFrame([x for x in F], columns=["X%s" % k for k in range(F.shape[1])])
    df["class"] = "Points"
    return pd.plotting.radviz(df, "class")
