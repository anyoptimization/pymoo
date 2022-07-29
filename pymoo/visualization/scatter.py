import numpy as np

from pymoo.core.plot import Plot
from pymoo.docs import parse_doc_string
from pymoo.util.misc import set_if_none


def plot_1d(sc):
    sc.init_figure()
    labels = sc.get_labels()
    ax = sc.ax

    for k, (F, kwargs) in enumerate(sc.to_plot):
        func = getattr(ax, kwargs.pop("mode"))
        func(F, np.zeros_like(F), **kwargs)
        ax.set_xlabel(labels[0])


def plot_2d(sc):
    sc.init_figure()
    labels = sc.get_labels()
    ax = sc.ax

    for k, (F, kwargs) in enumerate(sc.to_plot):
        func = getattr(ax, kwargs.pop("mode"))
        func(F[:, 0], F[:, 1], **kwargs)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    return sc


def plot_3d(sc, angle):
    sc.init_figure(plot_3D=True)
    labels = sc.get_labels()
    ax = sc.ax

    for k, (F, kwargs) in enumerate(sc.to_plot):

        # here alo `plot_trisurf` is allowed
        func = getattr(ax, kwargs.pop("mode"))
        func(F[:, 0], F[:, 1], F[:, 2], **kwargs)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])

        if sc.angle is not None:
            ax.view_init(*angle)


def plot_pairwise(sc):
    sc.init_figure(n_rows=sc.n_dim, n_cols=sc.n_dim)
    labels = sc.get_labels()

    for k, (F, kwargs) in enumerate(sc.to_plot):

        assert F.shape[1] >= 2, "A pairwise sc plot needs at least two dimensions."
        mode = kwargs.pop("mode")

        for i in range(sc.n_dim):
            for j in range(sc.n_dim):

                ax = sc.ax[i, j]
                func = getattr(ax, mode)

                if i != j:
                    func(F[:, i], F[:, j], **kwargs)
                    ax.set_xlabel(labels[i])
                    ax.set_ylabel(labels[j])
                else:
                    func(0, 0, s=1, color="white")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(0, 0, labels[i], ha='center', va='center', fontsize=20)


class Scatter(Plot):

    def __init__(self,
                 plot_3d=True,
                 angle=(45, 45),
                 **kwargs):
        """

        Scatter Plot

        Parameters
        ----------------

        axis_style : {axis_style}
        labels : {labels}

        Other Parameters
        ----------------

        figsize : {figsize}
        title : {title}
        legend : {legend}
        tight_layout : {tight_layout}

        """

        super().__init__(**kwargs)
        self.angle = angle
        self.plot_3d = plot_3d

    def _do(self):

        # set some default values
        to_plot = []
        for k, (F, v) in enumerate(self.to_plot):
            v = dict(v)
            set_if_none(v, "color", self.colors[k % len(self.colors)])
            set_if_none(v, "alpha", 1.0)

            # this is added to have compatibility to an old version
            # should be removed when the documentation is updated
            if "plot_type" in v:
                name = v.pop("plot_type")

                if name == "line":
                    name = "plot"
                elif name == "surface":
                    name = "plot_trisurf"

                v["mode"] = name
            set_if_none(v, "mode", "scatter")

            to_plot.append([F, v])

        self.to_plot = to_plot

        if self.n_dim == 1:
            plot_1d(self)
        elif self.n_dim == 2:
            plot_2d(self)
        elif self.n_dim == 3 and self.plot_3d:
            plot_3d(self, self.angle)
        else:
            plot_pairwise(self)

        return self


parse_doc_string(Scatter.__init__)
