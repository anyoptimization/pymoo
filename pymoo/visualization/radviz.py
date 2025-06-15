from pymoo.docs import parse_doc_string
from pymoo.core.plot import Plot
from pymoo.util.misc import set_if_none_from_tuples
from pymoo.visualization.util import plot_circle, plot_radar_line, plot_axis_labels, equal_axis, no_ticks, \
    get_uniform_points_around_circle


class Radviz(Plot):

    def __init__(self, endpoint_style={}, **kwargs):
        """

        Radviz Plot

        Parameters
        ----------

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

        """

        super().__init__(**kwargs)

        # set the default axis style
        set_if_none_from_tuples(self.axis_style, ("color", "black"), ("linewidth", 1), ("alpha", 0.75))

        self.endpoint_style = endpoint_style
        set_if_none_from_tuples(self.endpoint_style, ("color", "black"), ("s", 70), ("alpha", 0.3))

    def _do(self):

        # initial a figure with a single plot
        self.init_figure()

        # equal axis length and no ticks
        equal_axis(self.ax)
        no_ticks(self.ax)

        V = get_uniform_points_around_circle(self.n_dim)
        plot_axis_labels(self.ax, V, self.get_labels(), **self.axis_label_style)

        # draw the outer circle and radar lines
        plot_circle(self.ax, **self.axis_style)
        plot_radar_line(self.ax, V, **self.axis_style)

        # draw the endpoints of each objective
        if self.endpoint_style:
            self.ax.scatter(V[:, 0], V[:, 1], **self.endpoint_style)

        # plot all the points
        for k, (F, kwargs) in enumerate(self.to_plot):
            N = (F[..., None] * V).sum(axis=1) / F.sum(axis=1)[:, None]
            self.ax.scatter(N[:, 0], N[:, 1], **kwargs)


parse_doc_string(Radviz.__init__)
