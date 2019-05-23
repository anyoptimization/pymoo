import numpy as np
from matplotlib.patches import Circle

from pymoo.analytics.visualization.plot import Plot
from pymoo.operators.default_operators import set_if_none


class ProjectionPlot(Plot):

    def __init__(self, **kwargs):
        super().__init__(figsize=(6, 6), **kwargs)
        self.V = None

    def draw_axes(self):
        extend_factor = 1.03
        for x, y in self.V:
            self.ax.arrow(0, 0, x * extend_factor, y * extend_factor, **self.axes_style)
        self.ax.add_artist(Circle([0, 0], radius=0.01, color=self.axes_style["color"], alpha=0.5))

    def draw_axis_labels(self, labels):

        for k in range(len(labels)):
            xy = self.V[k]
            margin = 0.035
            if xy[0] < 0.0 and xy[1] < 0.0:
                self.ax.text(xy[0] - margin, xy[1] - margin, labels[k], ha='right', va='top', size='small')
            elif xy[0] < 0.0 and xy[1] >= 0.0:
                self.ax.text(xy[0] - margin, xy[1] + margin, labels[k], ha='right', va='bottom', size='small')
            elif xy[0] >= 0.0 and xy[1] < 0.0:
                self.ax.text(xy[0] + margin, xy[1] - margin, labels[k], ha='left', va='top', size='small')
            elif xy[0] >= 0.0 and xy[1] >= 0.0:
                self.ax.text(xy[0] + margin, xy[1] + margin, labels[k], ha='left', va='bottom', size='small')

    def _do(self):
        _, n_obj = self.to_plot[0][0].shape

        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlim([-1.1, 1.1])
        self.ax.set_ylim([-1.1, 1.1])
        self.ax.axis('equal')

        t = 2 * np.pi * np.arange(n_obj) / n_obj
        self.V = np.column_stack([np.cos(t), np.sin(t)])

        self._plot()

        # Remove the ticks from the graph
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        self.ax.set_frame_on(False)

        return self

    def set_default(self, kwargs):
        set_if_none(kwargs, "s", 25)
        set_if_none(kwargs, "facecolors", 'none')
        set_if_none(kwargs, "alpha", 1.0)
