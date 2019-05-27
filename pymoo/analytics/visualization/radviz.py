from pymoo.analytics.visualization.projection import ProjectionPlot
from pymoo.analytics.visualization.util import plot_circle, plot_radar_line
from pymoo.docs import parse_doc_string


class Radviz(ProjectionPlot):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "endpoint_style" not in kwargs:
            self.endpoint_style = {
                "color": "black",
                "s": 70,
                "alpha": 0.3
            }
        else:
            self.endpoint_style = kwargs["endpoint_style"]

    def _plot(self):

        self.draw_axis_labels(self.get_labels(), self.V)

        # draw the outer circle and radar lines
        plot_circle(self.ax, **self.axis_style)
        plot_radar_line(self.ax, self.V, **self.axis_style)

        # draw the endpoints of each objective
        if self.endpoint_style:
            self.ax.scatter(self.V[:, 0], self.V[:, 1], **self.endpoint_style)

        # plot all the points
        for k, (F, kwargs) in enumerate(self.to_plot):
            N = (F[..., None] * self.V).sum(axis=1) / F.sum(axis=1)[:, None]
            self.ax.scatter(N[:, 0], N[:, 1], **kwargs)


# =========================================================================================================
# Interface
# =========================================================================================================


def radviz(**kwargs):
    """

    Radviz Plot

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
    Radviz : :class:`~pymoo.model.analytics.visualization.radviz.Radviz`

    """

    return Radviz(**kwargs)


parse_doc_string(radviz)
