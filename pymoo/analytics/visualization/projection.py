from pymoo.analytics.visualization.util import get_circle_points
from pymoo.model.plot import Plot
from pymoo.operators.default_operators import set_if_none


class ProjectionPlot(Plot):

    def __init__(self, **kwargs):
        super().__init__(figsize=(6, 6), **kwargs)
        self.V = None

        if self.axis_style is None:
            self.axis_style = {
                'color': 'grey',
                'linewidth': 1,
                'alpha': 0.8
            }


    def _do(self):
        _, n_obj = self.to_plot[0][0].shape

        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlim([-1.1, 1.1])
        self.ax.set_ylim([-1.1, 1.1])
        self.ax.axis('equal')

        self.V = get_circle_points(n_obj)
        self._plot()

        # Remove the ticks from the graph
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        self.ax.set_frame_on(False)

        return self
