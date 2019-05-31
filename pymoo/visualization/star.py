import numpy as np

from pymoo.visualization.util import plot_axes_arrow, plot_axis_labels, equal_axis, no_ticks, parse_bounds, \
    normalize, get_uniform_points_around_circle
from pymoo.docs import parse_doc_string
from pymoo.model.plot import Plot


class StarCoordinate(Plot):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.axis_extension = kwargs["axis_extension"]

        if "arrow_style" not in kwargs:
            self.arrow_style = {
                "head_width": 0.02,
                "head_length": 0.01
            }
        else:
            self.arrow_style = kwargs["arrow_style"]

    def _do(self):

        # initial a figure with a single plot
        self.init_figure()

        # equal axis length and no ticks
        equal_axis(self.ax)
        no_ticks(self.ax)

        # determine the overall scale of points
        _F = np.row_stack([e[0] for e in self.to_plot])
        _min, _max = _F.min(axis=0), _F.max(axis=0)

        V = get_uniform_points_around_circle(self.n_dim)

        plot_axes_arrow(self.ax, V, extend_factor=self.axis_extension, **{**self.axis_style, **self.arrow_style})
        plot_axis_labels(self.ax, V, self.get_labels())

        # normalize in range for this plot - here no implicit normalization as in radviz
        bounds = parse_bounds(self.bounds, self.n_dim)
        to_plot_norm = normalize(self.to_plot, bounds)

        for k, (F, kwargs) in enumerate(to_plot_norm):
            N = (F[..., None] * V).sum(axis=1)
            self.ax.scatter(N[:, 0], N[:, 1], **kwargs)


# =========================================================================================================
# Interface
# =========================================================================================================


def star_coordinate(axis_extension=1.03, **kwargs):
    """

    Star Coordinate Plot

    Parameters
    ----------------

    axis_style : {axis_style}

    endpoint_style : dict
        Endpoints are drawn at each extreme point of an objective. This style can be modified.
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
    StarCoordinate : :class:`~pymoo.model.analytics.visualization.star.StarCoordinate`

    """

    return StarCoordinate(axis_extension=axis_extension, **kwargs)


parse_doc_string(star_coordinate)
