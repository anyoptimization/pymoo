"""Radar plot visualization for multi-objective optimization results."""

import numpy as np

from pymoo.docs import parse_doc_string
from pymoo.core.plot import Plot
from pymoo.util.misc import set_if_none_from_tuples
from pymoo.visualization.util import (
    plot_axes_lines,
    plot_axis_labels,
    plot_polygon,
    get_circle_points,
    plot_radar_line,
    equal_axis,
    no_ticks,
    parse_bounds,
    normalize,
)


class Radar(Plot):
    """Radar plot visualization.

    Args:
        normalize_each_objective: Whether each objective is normalized. Otherwise, the inner and outer bound is plotted.
        n_partitions: Number of partitions to show in the radar.
        point_style: The style being used to visualize the points.
        reverse: {reverse}
        axis_style: {axis_style}
        labels: {labels}
        figsize: {figsize}
        title: {title}
        legend: {legend}
        tight_layout: {tight_layout}
        cmap: {cmap}
    """

    def __init__(
        self,
        normalize_each_objective: bool = True,
        n_partitions: int = 3,
        point_style: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.normalize_each_objective = normalize_each_objective
        self.n_partitions = n_partitions

        if point_style is None:
            point_style = {}
        self.point_style = point_style
        set_if_none_from_tuples(self.point_style, ("s", 15))

        set_if_none_from_tuples(
            self.axis_style, ("color", "black"), ("linewidth", 0.5), ("alpha", 0.75)
        )

    def _plot(
        self, ax, _F: np.ndarray, inner: np.ndarray, outer: np.ndarray, kwargs: dict
    ) -> None:
        set_if_none_from_tuples(kwargs, ("alpha", 0.5))

        # equal axis length and no ticks
        equal_axis(ax)
        no_ticks(ax)

        # draw the axis lines and labels
        plot_axes_lines(ax, outer, extend_factor=1.0, **self.axis_style)
        plot_axis_labels(
            ax, outer, self.get_labels(), margin=0.015, **self.axis_label_style
        )

        # plot the outer radar line and the inner polygon
        plot_radar_line(ax, outer, **self.axis_style)
        plot_polygon(ax, inner)

        # find the corresponding point
        _F = inner + _F[:, None] * (outer - inner)

        # plot the points and no polygon
        ax.scatter(_F[:, 0], _F[:, 1], **self.point_style)
        plot_polygon(ax, _F, **kwargs)

    def _do(self) -> None:
        if self.bounds is None:
            raise Exception("The boundaries must be provided.")

        _F = np.vstack([e[0] for e in self.to_plot])
        if np.any(_F < self.bounds[0]) or np.any(_F > self.bounds[1]):
            raise Exception(
                "Points out of the boundaries exist! Please make sure the boundaries are indeed boundaries."
            )

        n_rows = len(self.to_plot)
        n_cols = max([len(e[0]) for e in self.to_plot])
        self.init_figure(n_rows=n_rows, n_cols=n_cols, force_axes_as_matrix=True)

        # normalize the input
        bounds = parse_bounds(self.bounds, self.n_dim)
        to_plot_norm = normalize(self.to_plot, bounds, reverse=self.reverse)

        # get the endpoints of circle
        V = get_circle_points(self.n_dim)

        if self.normalize_each_objective:
            inner = np.zeros((self.n_dim, 1)) * V
            outer = np.ones((self.n_dim, 1)) * V
        else:
            inner = bounds[[0]].T * V
            outer = (bounds[[1]].T * V) / bounds[1].max()

        for k, (F, kwargs) in enumerate(to_plot_norm):
            for j, _F in enumerate(F):
                self._plot(self.ax[k, j], _F, inner, outer, kwargs)


parse_doc_string(Radar.__init__)
