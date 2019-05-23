import numpy as np

from pymoo.analytics.visualization.star import StarCoordinate
from pymoo.analytics.visualization.util import plot_radar_line
from pymoo.operators.default_operators import set_if_none


class Radar(StarCoordinate):

    def __init__(self, n_partitions=5, **kwargs):
        super().__init__(**kwargs)
        self.n_partitions = n_partitions

    def _plot(self):

        # plot the lines from each objective point
        lines = np.row_stack([self.V, self.V[0]])

        for e in np.linspace(0, 1, self.n_partitions):
            self.ax.plot(lines[:, 0] * e, lines[:, 1] * e, color="grey", alpha=0.5)
            # self.ax.scatter(self.V[:, 0] * e, self.V[:, 1] * e, color="grey", alpha=0.5)

        self.draw_axes()
        self.draw_axis_labels(self.get_labels())

        # normalize in range
        self.parse_bounds()
        self.normalize()

        # plot all the points
        for k, (F, kwargs) in enumerate(self.to_plot):

            set_if_none(kwargs, "color", self.cmap[k])

            for k, F in enumerate(F):
                N = (1 - F[:, None]) * self.V

                kwargs["alpha"] = 0.95
                self.ax.scatter(N[:, 0], N[:, 1], **kwargs)

                kwargs["alpha"] = 0.35
                kwargs["label"] = None
                plot_radar_line(self.ax, N, kwargs, filled=True)




if __name__ == "__main__":
    np.random.seed(2)

    # Radviz().add(np.random.random((1000, 7))).show()
    # Radviz().add(sample_on_simplex(10000,7)).show()

    X = np.random.random((1, 5))
    print(X)

    Radar(legend=True, bounds=[0,1]) \
        .add(np.array([0.1, 0.1, 0.1, 0.1, 0.1]), label="First") \
        .add(np.array([0.6, 0.4, 0.3, 0.2, 0.6]), label="second") \
        .show()
    # Radviz().add(np.eye(5)[[0], :]).show()
