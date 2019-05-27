import matplotlib.pyplot as plt
import numpy as np

from pymoo.docs import parse_doc_string
from pymoo.model.plot import Plot
from pymoo.operators.default_operators import set_if_none


class ParallelCoordinatePlot(Plot):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.show_bounds = kwargs["show_bounds"]
        self.n_ticks = kwargs["n_ticks"]
        self.normalize_each_axis = kwargs["normalize_each_axis"]

        set_if_none(self.axis_style, "color", "red")
        set_if_none(self.axis_style, "linewidth", 2)
        set_if_none(self.axis_style, "alpha", 0.8)

    def _do(self):
        self.ax = self.fig.add_subplot(1, 1, 1)

        # if no normalization of each axis the bounds are based on the overall min and max
        if not self.normalize_each_axis and self.bounds is None:
            _F = np.row_stack([e[0] for e in self.to_plot])
            self.bounds = [_F.min(), _F.max()]

        self.parse_bounds()
        to_plot = self.normalize()

        # plot for each set the lines
        for k, (F, kwargs) in enumerate(to_plot):
            set_if_none(kwargs, "color", self.colors[k])

            for i in range(len(F)):
                plt.plot(np.arange(F.shape[1]), F[i, :], **kwargs)

        # Plot the parallel coordinate axes
        for i in range(self.n_dim):
            self.ax.axvline(i, **self.axis_style)

            bottom, top = -0.1, 1.075
            margin_left = 0.08

            if self.show_bounds:
                self.ax.text(i - margin_left, bottom, self.func_number_to_text(self.bounds[0][i]))
                self.ax.text(i - margin_left, top, self.func_number_to_text(self.bounds[1][i]))

            if self.n_ticks is not None:
                n_length = 0.03
                for y in np.linspace(0, 1, self.n_ticks):
                    self.ax.hlines(y, i - n_length, i + n_length, **self.axis_style)

        # if bounds are shown, then move them to the bottom
        if self.show_bounds:
            self.ax.tick_params(axis='x', which='major', pad=25)

        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)

        self.ax.set_yticklabels([])
        self.ax.set_yticks([])
        self.ax.set_ylim((-0.05, 1.05))

        self.ax.set_xticks(np.arange(self.n_dim))
        self.ax.set_xticklabels(self.get_labels())

        return self


# =========================================================================================================
# Interface
# =========================================================================================================


def pcp(bounds=None,
        show_bounds=True,
        n_ticks=5,
        normalize_each_axis=True,
        **kwargs):
    """

    Parallel Coordinate Plot


    Parameters
    ----------------

    bounds : {bounds}

    axis_style : {axis_style}

    labels : {labels}

    n_ticks : int
        Number of ticks to be shown on each parallel axis.

    show_bounds : bool
        Whether the value of the boundaries are shown in the plot or not.

    normalize_each_axis : bool
        Whether the values should be normalized either by bounds or implictly.

    Other Parameters
    ----------------

    figsize : {figsize}
    title : {title}
    legend : {legend}
    tight_layout : {tight_layout}
    cmap : {cmap}



    Returns
    -------
    ParallelCoordinatePlot : :class:`~pymoo.model.analytics.visualization.ParallelCoordinatePlot`

    """

    return ParallelCoordinatePlot(bounds=bounds,
                                  show_bounds=show_bounds,
                                  n_ticks=n_ticks,
                                  normalize_each_axis=normalize_each_axis,
                                  **kwargs)


parse_doc_string(pcp)
