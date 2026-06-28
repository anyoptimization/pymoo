"""Heatmap visualization for multi-objective optimization results."""

import numpy as np

from pymoo.core.plot import Plot
from pymoo.util.misc import set_if_none_from_tuples
from pymoo.visualization.util import normalize, parse_bounds


class Heatmap(Plot):
    """Visualize results as a heatmap."""

    def __init__(
        self,
        cmap="Blues",
        order_by_objectives=False,
        reverse=True,
        solution_labels=True,
        **kwargs,
    ):
        """Initialize Heatmap.

        Args:
            cmap: The color map to be used.
            order_by_objectives: Whether the result should be ordered by an objective.
                If False no order. Otherwise, either supply just the objective or a list
                (it is lexicographically sorted).
            reverse: If True large values are white and small values the corresponding color.
                Otherwise, the other way around.
            solution_labels: If False no labels are plotted in the y axis. If True just the
                corresponding index. Otherwise, the label provided.
            **kwargs: Additional keyword arguments passed to parent Plot class.
        """
        super().__init__(cmap=cmap, reverse=reverse, **kwargs)
        self.order_by_objectives = order_by_objectives
        self.solution_labels = solution_labels

        # set default style
        set_if_none_from_tuples(self.axis_style, ("interpolation", "nearest"), ("vmin", 0), ("vmax", 1))

    def _do(self):

        if len(self.to_plot) != 1:
            raise Exception("Only one element can be added to a heatmap.")

        # initial a figure with a single plot
        self.init_figure()

        # normalize the input
        bounds = parse_bounds(self.bounds, self.n_dim)
        to_plot_norm = normalize(self.to_plot, bounds, reverse=self.reverse)
        (F, kwargs) = to_plot_norm[0]

        # dot the sorting if required
        if self.order_by_objectives is not None and self.order_by_objectives is not False:
            if isinstance(self.order_by_objectives, list) and len(self.order_by_objectives) == self.n_dim:
                L = self.order_by_objectives
            elif isinstance(self.order_by_objectives, int):
                L = [i for i in range(F.shape[1]) if i != self.order_by_objectives]
                L.insert(0, self.order_by_objectives)
            else:
                L = range(self.n_dim)

            _F = [F[:, j] for j in L]
            indices = np.lexsort(_F[::-1])
        else:
            indices = np.arange(len(F))

        # plot the data
        self.ax.imshow(F[indices], cmap=self.cmap, **self.axis_style)

        # set the x ticks and labels
        self.ax.set_xticks(np.arange(self.n_dim))
        self.ax.set_xticklabels(self.get_labels())

        # no solution labels should be used
        if self.solution_labels is None:
            pass

        # if true, just use a number for each solution
        elif isinstance(self.solution_labels, bool) and self.solution_labels:
            self.solution_labels = np.arange(len(F)) + 1

        # otherwise, use directly the label provided
        else:
            if len(self.solution_labels) != len(F):
                raise Exception(
                    "The labels provided for each solution must be equal to the number of solutions being plotted."
                )

        if self.solution_labels is None:
            self.ax.set_yticks([])
            self.ax.set_yticklabels([])

        else:
            # for ordered by objective apply it to labels
            self.solution_labels = [self.solution_labels[i] for i in indices]

            self.ax.set_yticks(np.arange(len(F)))
            self.ax.set_yticklabels(self.solution_labels)
