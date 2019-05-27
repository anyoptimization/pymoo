import numpy as np

from pymoo.docs import parse_doc_string
from pymoo.model.plot import Plot
from pymoo.operators.default_operators import set_if_none


class Heatmap(Plot):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order_by_objectives = kwargs["order_by_objectives"]
        self.y_labels = kwargs["y_labels"]

    def _do(self):
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.normalize()

        if len(self.to_plot) != 1:
            raise Exception("Only one element can be added to a heatmap.")

        (F, kwargs) = self.to_plot[0]

        if self.order_by_objectives is not None and self.order_by_objectives is not False:

            if isinstance(self.order_by_objectives, list) and len(self.order_by_objectives) == self.n_dim:
                L = self.order_by_objectives
            elif isinstance(self.order_by_objectives, int):
                L = [i for i in range(F.shape[1]) if i != self.order_by_objectives]
                L.insert(0, self.order_by_objectives)
            else:
                L = range(self.n_dim)

            _F = [F[:, j] for j in L]
            I = np.lexsort(_F[::-1])
        else:
            I = np.arange(len(F))

        set_if_none(kwargs, "interpolation", "nearest")
        set_if_none(kwargs, "vmin", 0)
        set_if_none(kwargs, "vmax", 1)

        if self.reverse:
            F = 1 - F

        self.ax.imshow(F[I], cmap=self.cmap, **kwargs)

        self.ax.set_xticks(np.arange(self.n_dim))
        self.ax.set_xticklabels(self.get_labels())

        if self.y_labels is None:
            self.y_labels = ["" for _ in range(len(F))]
        elif isinstance(self.y_labels, bool) and self.y_labels:
            self.y_labels = np.arange(len(F)) + 1
        else:
            if len(self.y_labels) != len(F):
                raise Exception(
                    "The labels provided for each solution must be equal to the number of solutions being plotted.")

        self.y_labels = [self.y_labels[i] for i in I]

        self.ax.set_yticks(np.arange(len(F)))
        self.ax.set_yticklabels(self.y_labels)


# =========================================================================================================
# Interface
# =========================================================================================================


def heatmap(cmap="Blues",
            order_by_objectives=False,
            reverse=True,
            y_labels=True,
            **kwargs):
    """

    Heatmap

    Parameters
    ----------------

    cmap : str
        The color map to be used.

    order_by_objectives : int or list
        Whether the result should be ordered by an objective. If false no order.
        Otherwise, either supply just the objective or a list. (it is lexicographically sorted).

    reverse : bool
        If true large values are white and small values the corresponding color. Otherwise, the other way around.

    y_labels : bool or list
        If False no labels are plotted in the y axis. If true just the corresponding index. Otherwise the label provided.

    bounds : {bounds}

    labels : {labels}

    Other Parameters
    ----------------

    figsize : {figsize}
    title : {title}
    legend : {legend}
    tight_layout : {tight_layout}
    cmap : {cmap}


    Returns
    -------
    Heatmap : :class:`~pymoo.model.analytics.visualization.heatmap.Heatmap`

    """

    return Heatmap(cmap=cmap, order_by_objectives=order_by_objectives, reverse=reverse, y_labels=y_labels, **kwargs)


parse_doc_string(heatmap)
