"""Petal diagram visualization for multi-objective optimization."""

import numpy as np

from pymoo.core.plot import Plot
from pymoo.util.misc import set_if_none
from pymoo.visualization.util import (
    equal_axis,
    get_circle_points,
    no_ticks,
    normalize,
    parse_bounds,
    plot_axes_lines,
    plot_axis_labels,
    plot_circle,
    plot_polygon,
)


class Petal(Plot):
    """Petal diagram for multi-objective visualization."""

    def __init__(
        self,
        bounds=None,
        **kwargs,
    ):
        """Initialize Petal diagram.

        Args:
            bounds: The boundaries for each objective. Necessary to be provided for this plot!
            **kwargs: Additional keyword arguments passed to parent Plot class.
        """
        super().__init__(bounds=bounds, **kwargs)

        if bounds is None:
            raise Exception(
                "Boundaries must be provided for Petal Width. Otherwise, no trade-offs can be calculated."
            )

        set_if_none(self.axis_style, "color", "black")
        set_if_none(self.axis_style, "linewidth", 2)
        set_if_none(self.axis_style, "alpha", 0.5)

    def _plot(self, ax, F):

        # equal axis length and no ticks
        equal_axis(ax)
        no_ticks(ax)

        V = get_circle_points(len(F))

        # sections to plot
        sections = np.linspace(0, 2 * np.pi, self.n_dim + 1)

        t = [(sections[i] + sections[i + 1]) / 2 for i in range(len(sections) - 1)]
        endpoints = np.column_stack([np.cos(t), np.sin(t)])
        plot_axis_labels(ax, endpoints, self.get_labels(), **self.axis_label_style)

        center = np.zeros(2)

        for i in range(len(sections) - 1):
            t = np.linspace(sections[i], sections[i + 1], 100)
            v = np.column_stack([np.cos(t), np.sin(t)])

            P = np.vstack([center, F[i] * v])
            plot_polygon(ax, P, color=self.colors[i])

        # draw the outer circle
        plot_circle(ax, **self.axis_style)
        plot_axes_lines(ax, V, **self.axis_style)

    def _do(self):

        n_rows = len(self.to_plot)
        n_cols = max([len(e[0]) for e in self.to_plot])
        self.init_figure(n_rows=n_rows, n_cols=n_cols, force_axes_as_matrix=True)

        # normalize the input
        bounds = parse_bounds(self.bounds, self.n_dim)
        to_plot_norm = normalize(self.to_plot, bounds, reverse=self.reverse)

        for k, (F, kwargs) in enumerate(to_plot_norm):
            for j, _F in enumerate(F):
                self._plot(self.ax[k, j], _F)
