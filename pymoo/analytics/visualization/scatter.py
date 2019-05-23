import numpy as np

from pymoo.analytics.visualization.plot import Plot
from pymoo.operators.default_operators import set_if_none


class Scatter(Plot):

    def __init__(self,
                 angle=None,
                 **kwargs):

        super().__init__(**kwargs)
        self.angle = angle

    def _do(self):
        is_3d = (self.n_dim == 3)

        # create the axis object
        if not is_3d:
            self.ax = self.fig.add_subplot(1, 1, 1)
        else:
            self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')

        # set the labels for each axis
        labels = self.get_labels()
        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])
        if is_3d:
            self.ax.set_zlabel(labels[2])

        # plot for each entry
        for k, (F, kwargs) in enumerate(self.to_plot):

            self.set_default(kwargs)
            set_if_none(kwargs, "color", self.cmap[k])

            _type = kwargs.get("plot_type")

            if "plot_type" in kwargs:
                _type = kwargs["plot_type"]
                del kwargs["plot_type"]
            else:
                _type = "scatter"

            if self.n_dim == 1:
                raise Exception("1D Interval not implemented yet.")

            elif self.n_dim == 2:
                if _type == "scatter":
                    self.ax.scatter(F[:, 0], F[:, 1], **kwargs)
                else:
                    self.ax.plot(F[:, 0], F[:, 1], **kwargs)

            elif self.n_dim == 3:
                self.ax.scatter(F[:, 0], F[:, 1], F[:, 2], **kwargs)
                self.ax.xaxis.pane.fill = False
                self.ax.yaxis.pane.fill = False
                self.ax.zaxis.pane.fill = False
            else:
                raise Exception("Scatter Pair Plots not implemented yet.")

        # change the angle for 3d plots
        if is_3d and self.angle is not None:
            self.ax.view_init(45, 45)

        if self.legend:
            self.ax.legend()

        return self

    def set_default(self, kwargs):
        #set_if_none(kwargs, "s", 25)
        # set_if_none(kwargs, "facecolors", 'none')
        set_if_none(kwargs, "alpha", 1.0)


if __name__ == "__main__":
    np.random.seed(1)

    Scatter(tight_layout=True, legend=False, angle=(45, 45)) \
        .add(np.random.random((10, 3)), facecolors="blue", label="Test") \
        .add(np.random.random((10, 3)), facecolors="red", label="Blue") \
        .do() \
        .apply(lambda ax: ax.legend()) \
        .show()
