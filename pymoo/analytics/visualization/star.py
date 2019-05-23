import numpy as np

from pymoo.analytics.visualization.projection import ProjectionPlot
from pymoo.operators.default_operators import set_if_none
from pymoo.problems.util import UniformReferenceDirectionFactory
from pymoo.util.normalization import normalize


class StarCoordinate(ProjectionPlot):

    def __init__(self,
                 arrow_style={'color': 'grey', 'alpha': 0.85, 'head_width': 0.03, 'head_length': 0.01},
                 **kwargs):
        super().__init__(**kwargs)
        self.axes_style = arrow_style
        set_if_none(self.axes_style, "color", "black")

    def _plot(self):
        _F = np.row_stack([e[0] for e in self.to_plot])
        _min, _max = _F.min(axis=0), _F.max(axis=0)

        self.draw_axes()
        self.draw_axis_labels(self.get_labels())

        # normalize in range
        self.parse_bounds()
        self.normalize()

        for k, (F, kwargs) in enumerate(self.to_plot):
            N = (F[..., None] * self.V).sum(axis=1)
            self.ax.scatter(N[:, 0], N[:, 1], **kwargs)


if __name__ == "__main__":
    np.random.seed(1)
    X = UniformReferenceDirectionFactory(5, n_partitions=5).do()
    StarCoordinate().add(X).show()
