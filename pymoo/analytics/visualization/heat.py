import numpy as np

from pymoo.analytics.visualization.plot import Plot
from pymoo.operators.default_operators import set_if_none
from pymoo.util.normalization import normalize


class HeatMap(Plot):

    def __init__(self,
                 order_by_objectives=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.order_by_objectives = order_by_objectives

    def _do(self):
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.normalize()

        indices = []
        for k, (F, kwargs) in enumerate(self.to_plot):

            if self.order_by_objectives is not None:

                if isinstance(self.order_by_objectives, list) and len(self.order_by_objectives) == F.shape[1]:
                    L = self.order_by_objectives
                elif isinstance(self.order_by_objectives, int):
                    L = [i for i in range(F.shape[1]) if i != self.order_by_objectives]
                    L.insert(0, self.order_by_objectives)
                else:
                    L = range(F.shape[1])

                _F = [F[:, j] for j in L]
                I = np.lexsort(_F[::-1])
            else:
                I = np.arange(len(F))

            indices.extend(I)

            set_if_none(kwargs, "cmap", "Blues")
            set_if_none(kwargs, "interpolation", "nearest")
            set_if_none(kwargs, "vmin", 0)
            set_if_none(kwargs, "vmax", 1)

            self.ax.imshow(1 - F, **kwargs)

        self.ax.set_xticks(np.arange(self.n_dim))
        self.ax.set_xticklabels(self.get_labels())

        self.ax.set_yticks(np.arange(len(indices)))
        self.ax.set_yticklabels(np.array(indices) + 1)


if __name__ == "__main__":
    np.random.seed(2)

    # Radviz().add(np.random.random((1000, 7))).show()
    # Radviz().add(sample_on_simplex(10000,7)).show()

    X = np.random.random((10, 7)) * 10
    print(X)

    HeatMap(legend=False, order_by_objectives=3) \
        .add(X) \
        .show()
    # Radviz().add(np.eye(5)[[0], :]).show()
