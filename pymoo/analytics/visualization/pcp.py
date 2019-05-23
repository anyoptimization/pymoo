import matplotlib.pyplot as plt
import numpy as np

from pymoo.analytics.visualization.plot import Plot, number_to_text
from pymoo.operators.default_operators import set_if_none
from pymoo.problems.util import UniformReferenceDirectionFactory


class ParallelCoordinatePlot(Plot):

    def __init__(self,
                 bounds=None,
                 show_bounds=True,
                 number_to_text=number_to_text,
                 normalize_each_axis=True,
                 axis_style={'color': "blue"},
                 show_ticks=True,
                 **kwargs):

        super().__init__(**kwargs)
        self.show_bounds = show_bounds
        self.show_ticks = show_ticks
        self.axis_style = axis_style
        self.number_to_text = number_to_text
        self.normalize_each_axis = normalize_each_axis
        self.bounds = bounds

    def _do(self):
        self.ax = self.fig.add_subplot(1, 1, 1)

        if not self.normalize_each_axis:
            _F = np.row_stack([e[0] for e in self.to_plot])
            self.bounds = [_F.min(), _F.max()]

        self.parse_bounds()
        self.normalize()

        for k, (F, kwargs) in enumerate(self.to_plot):
            set_if_none(kwargs, "color", self.cmap[k])

            for i in range(len(F)):
                plt.plot(np.arange(F.shape[1]), F[i, :], **kwargs)

        for i in range(self.n_dim):
            set_if_none(self.axis_style, "linewidth", 2)
            set_if_none(self.axis_style, "alpha", 0.8)
            self.ax.axvline(i, **self.axis_style)

            bottom, top = -0.1, 1.075
            margin_left = 0.08

            if self.show_bounds:
                self.ax.text(i - margin_left, bottom, self.number_to_text(self.bounds[0][i]))
                self.ax.text(i - margin_left, top, self.number_to_text(self.bounds[1][i]))

            if self.show_ticks:
                n_length = 0.03
                for y in np.linspace(0, 1, 10):
                    self.ax.hlines(y, i - n_length, i + n_length, **self.axis_style)

        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)

        self.ax.set_yticklabels([])
        self.ax.set_yticks([])
        self.ax.set_ylim((-0.05, 1.05))

        self.ax.set_xticks(np.arange(self.n_dim))
        self.ax.set_xticklabels(self.get_labels())

        if self.show_bounds:
            self.ax.tick_params(axis='x', which='major', pad=25)

        return self


if __name__ == "__main__":
    np.random.seed(1)

    X = UniformReferenceDirectionFactory(7, n_partitions=5).do()
    X[:, 2] *= 10

    ParallelCoordinatePlot(normalize_each_axis=True, bounds=[0.5, 1]) \
        .add(X) \
        .add(X[100], color="red", linewidth=3) \
        .show()
