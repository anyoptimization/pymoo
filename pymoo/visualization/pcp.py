import numpy as np

import pymoo.visualization.util
from pymoo.docs import parse_doc_string
from pymoo.core.plot import Plot
from pymoo.util.misc import set_if_none, set_if_none_from_tuples
from pymoo.visualization.util import parse_bounds, normalize


class PCP(Plot):

    def __init__(self,
                 bounds=None,
                 show_bounds=True,
                 n_ticks=5,
                 normalize_each_axis=True,
                 bbox=False,
                 **kwargs):
        """

        Parallel Coordinate Plot


        Parameters
        ----------

        bounds : {bounds}

        axis_style : {axis_style}

        labels : {labels}

        n_ticks : int
            Number of ticks to be shown on each parallel axis.

        show_bounds : bool
            Whether the value of the boundaries are shown in the plot or not.

        normalize_each_axis : bool
            Whether the values should be normalized either by bounds or implicitly.

        Other Parameters
        ----------------

        figsize : {figsize}
        title : {title}
        legend : {legend}
        tight_layout : {tight_layout}
        cmap : {cmap}

        """

        super().__init__(bounds=bounds, **kwargs)
        self.show_bounds = show_bounds
        self.n_ticks = n_ticks
        self.bbox = bbox
        self.normalize_each_axis = normalize_each_axis

        set_if_none_from_tuples(self.axis_style, ("color", "red"), ("linewidth", 2), ("alpha", 0.75))

    def _do(self):

        # initial a figure with a single plot
        self.init_figure()

        # if no normalization of each axis the bounds are based on the overall min and max
        if not self.normalize_each_axis and self.bounds is None:
            _F = np.vstack([e[0] for e in self.to_plot])
            self.bounds = [_F.min(), _F.max()]

        # normalize the input
        bounds = parse_bounds(self.bounds, self.n_dim)
        to_plot_norm, bounds = normalize(self.to_plot, bounds, return_bounds=True)

        # plot for each set the lines
        for k, (F, kwargs) in enumerate(to_plot_norm):

            _kwargs = kwargs.copy()
            set_if_none(_kwargs, "color", self.colors[k % len(self.colors)])

            for i in range(len(F)):
                pymoo.visualization.util.plot(np.arange(F.shape[1]), F[i, :], **_kwargs)

        # Plot the parallel coordinate axes
        for i in range(self.n_dim):
            self.ax.axvline(i, **self.axis_style)

            bottom, top = -0.1, 1.075
            margin_left = 0.08

            if self.show_bounds:
                lower = self.ax.text(i - margin_left, bottom, self.func_number_to_text(bounds[0][i]))
                upper = self.ax.text(i - margin_left, top, self.func_number_to_text(bounds[1][i]))

                if self.bbox:
                    lower.set_bbox(dict(facecolor='white', alpha=0.8))
                    upper.set_bbox(dict(facecolor='white', alpha=0.8))

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


parse_doc_string(PCP.__init__)
