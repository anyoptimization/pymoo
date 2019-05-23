import numpy as np


def get_uniform_points_around_circle(n):
    t = 2 * np.pi * np.arange(n) / n
    s = np.column_stack([np.cos(t), np.sin(t)])
    return s


def plot_radar_line(ax, x, kwargs, filled=False):
    lines = np.row_stack([x, x[0]])

    if not filled:
        ax.plot(lines[:, 0], lines[:, 1], **kwargs)
    else:
        ax.fill_between(lines[:, 0], lines[:, 1], **kwargs)


def normalize(F, bounds):
    pass
