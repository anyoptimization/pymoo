import matplotlib.pyplot as plt
import numpy as np

from pymoo.analytics.visualization.util import plot_axes_lines, plot_axis_labels, plot_polygon, get_circle_points, \
    plot_radar_line
from pymoo.docs import parse_doc_string
from pymoo.model.plot import Plot
from pymoo.operators.default_operators import set_if_none


class Radar(Plot):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.normalize_each_objective = kwargs["normalize_each_objective"]
        self.n_partitions = kwargs["n_partitions"]

        self.polygon_style = kwargs["polygon_style"]
        set_if_none(self.polygon_style, "alpha", 0.5)

        self.point_style = kwargs["point_style"]
        set_if_none(self.point_style, "s", 15)

        set_if_none(self.axis_style, "color", "black")
        set_if_none(self.axis_style, "linewidth", 0.5)
        set_if_none(self.axis_style, "alpha", 0.8)

    def _plot(self, ax, _F, inner, outer, kwargs):

        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.axis('equal')

        plot_axes_lines(ax, outer, extend_factor=1.0, **self.axis_style)
        plot_axis_labels(ax, outer, self.get_labels(), margin=0.015)

        plot_radar_line(ax, outer, **self.axis_style)
        plot_polygon(ax, inner)

        _F = inner + _F[:, None] * (outer - inner)

        kwargs["alpha"] = 0.95
        ax.scatter(_F[:, 0], _F[:, 1], **self.point_style)

        kwargs["alpha"] = 0.35
        kwargs["label"] = None
        plot_polygon(ax, _F, **self.polygon_style)

        # Remove the ticks from the graph
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_frame_on(False)

    def _do(self):

        n_rows = len(self.to_plot)
        n_cols = max([len(e[0]) for e in self.to_plot])
        self.fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=self.figsize)
        axes = np.array(axes).reshape(n_rows, n_cols)

        self.parse_bounds()
        to_plot = self.normalize()

        self.V = get_circle_points(self.n_dim)

        if self.bounds is None:
            raise Exception("The boundaries must be provided.")

        _F = np.row_stack([e[0] for e in self.to_plot])
        if np.any(_F < self.bounds[0]) or np.any(_F > self.bounds[1]):
            raise Exception(
                "Points out of the boundaries exist! Please make sure the boundaries are indeed boundaries.")

        if self.normalize_each_objective:
            inner = np.zeros((self.n_dim, 1)) * self.V
            outer = np.ones((self.n_dim, 1)) * self.V
        else:
            inner = self.bounds[[0]].T * self.V
            outer = (self.bounds[[1]].T * self.V) / self.bounds[1].max()

        for k, (F, kwargs) in enumerate(to_plot):

            if self.reverse:
                F = 1 - F

            for j, _F in enumerate(F):
                self._plot(axes[k, j], _F, inner, outer, kwargs)


# =========================================================================================================
# Interface
# =========================================================================================================


def radar(normalize_each_objective=True,
          n_partitions=3,
          polygon_style={},
          point_style={},
          **kwargs):
    """

    Radar Plot

    Parameters
    ----------------
    normalize_each_objective : bool
        Whether each objective is normalized. Otherwise the inner and outer bound is plotted.

    polygon_style : dict
        The style being used for the polygon

    n_partitions : int
        Number of partitions to show in the radar.

    axis_style : {axis_style}
    labels : {labels}

    Other Parameters
    ----------------

    figsize : {figsize}
    title : {title}
    legend : {legend}
    tight_layout : {tight_layout}
    cmap : {cmap}


    Returns
    -------
    Radar : :class:`~pymoo.model.analytics.visualization.radar.Radar`

    """

    return Radar(normalize_each_objective=normalize_each_objective,
                 n_partitions=n_partitions,
                 polygon_style=polygon_style,
                 point_style=point_style,
                 **kwargs)


parse_doc_string(radar)
