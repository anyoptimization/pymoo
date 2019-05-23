import numpy as np
from matplotlib import patches

from pymoo.analytics.visualization.projection import ProjectionPlot
from pymoo.analytics.visualization.util import plot_radar_line
from pymoo.problems.util import UniformReferenceDirectionFactory


class Radviz(ProjectionPlot):
    """
    Inspired by https://github.com/DistrictDataLabs/yellowbrick/blob/master/yellowbrick/features/radviz.py
    """

    def _plot(self):

        self.draw_axis_labels(self.get_labels())

        # draw the outer circle
        self.ax.add_patch(patches.Circle((0.0, 0.0), radius=1.0, facecolor='none', edgecolor='grey', linewidth=.5))

        # plot the lines from each objective point
        plot_radar_line(self.ax, self.V, {'color': "grey"})
        self.ax.scatter(self.V[:, 0], self.V[:, 1], color="black")

        # plot all the points
        for k, (F, kwargs) in enumerate(self.to_plot):
            N = (F[..., None] * self.V).sum(axis=1) / F.sum(axis=1)[:, None]
            self.ax.scatter(N[:, 0], N[:, 1], **kwargs)


def radviz_pandas(F):
    import pandas as pd
    df = pd.DataFrame([x for x in F], columns=["X%s" % k for k in range(F.shape[1])])
    df["class"] = "Points"
    return pd.plotting.radviz(df, "class")


if __name__ == "__main__":
    np.random.seed(1)

    # Radviz().add(np.random.random((1000, 7))).show()
    # Radviz().add(sample_on_simplex(10000,7)).show()

    X = UniformReferenceDirectionFactory(6, n_partitions=5).do()

    Radviz().add(X).save("test.pdf").show()
    # Radviz().add(np.eye(5)[[0], :]).show()
