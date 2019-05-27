import numpy as np

from pymoo.analytics.visualization.projection import ProjectionPlot
from pymoo.analytics.visualization.util import plot_axes_arrow
from pymoo.docs import parse_doc_string


class StarCoordinate(ProjectionPlot):

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

    def _plot(self):
        _F = np.row_stack([e[0] for e in self.to_plot])
        _min, _max = _F.min(axis=0), _F.max(axis=0)

        _style = {**self.axis_style, **self.arrow_style}
        plot_axes_arrow(self.ax, self.V, extend_factor=self.axis_extension, **_style)
        self.draw_axis_labels(self.get_labels(), self.V)

        # normalize in range for this plot - here no implicit normalization as in radviz
        self.parse_bounds()
        to_plot = self.normalize()

        for k, (F, kwargs) in enumerate(to_plot):
            N = (F[..., None] * self.V).sum(axis=1)
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
