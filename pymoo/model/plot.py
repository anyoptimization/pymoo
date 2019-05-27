import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from pymoo.analytics.visualization.util import default_number_to_text, in_notebook
from pymoo.operators.default_operators import set_if_none
from pymoo.util.normalization import normalize


class Plot():

    def __init__(self,
                 figsize=(8, 6),
                 title=None,
                 legend=False,
                 tight_layout=False,
                 bounds=None,
                 reverse=False,
                 cmap="tab10",
                 axis_style={},
                 func_number_to_text=default_number_to_text,
                 labels="f",
                 **kwargs):

        super().__init__()
        self.figsize = figsize
        self.fig = plt.figure(figsize=figsize)

        self.to_plot = []
        self.ax = None
        self.legend = legend
        self.tight_layout = tight_layout

        if isinstance(cmap, str):
            self.cmap = matplotlib.cm.get_cmap(cmap)
        else:
            self.cmap = cmap

        if isinstance(self.cmap, ListedColormap):
            self.colors = self.cmap.colors

        self.axis_style = axis_style

        self.func_number_to_text = func_number_to_text

        self.labels = labels
        self.title = title
        self.n_dim = None

        self.bounds = bounds
        self.reverse = reverse

        plt.rc('font', family='serif')
        # plt.rc('text', usetex=True)
        # plt.rc('xtick', labelsize='x-small')
        # plt.rc('ytick', labelsize='x-small')

    def do(self):

        if len(self.to_plot) == 0:
            raise Exception("No elements to plot were added yet.")

        unique_dim = np.unique(np.array([e[0].shape[1] for e in self.to_plot]))
        if len(unique_dim) > 1:
            raise Exception("Inputs with different dimensions were added: %s" % unique_dim)

        self.n_dim = unique_dim[0]

        self._do()

        return self

    def parse_bounds(self):
        if self.bounds is not None:
            self.bounds = np.array(self.bounds, dtype=np.float)
            if self.bounds.ndim == 1:
                self.bounds = self.bounds[None, :].repeat(self.n_dim, axis=0).T

    def normalize(self):

        _F = np.row_stack([e[0] for e in self.to_plot])
        if self.bounds is None:
            self.bounds = (_F.min(axis=0), _F.max(axis=0))

        to_plot = []
        for k in range(len(self.to_plot)):
            F = normalize(self.to_plot[k][0], self.bounds[0], self.bounds[1])
            to_plot.append([F, self.to_plot[k][1]])

        return to_plot

    def apply(self, func):
        func(self.ax)
        return self

    def get_plot(self):
        return self.ax

    def set_axis_style(self, **kwargs):
        for key, val in kwargs.items():
            self.axis_style[key] = val
        return self

    def reset(self):
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = None
        return self

    def add(self, F, **kwargs):
        if F.ndim == 1:
            self.to_plot.append([F[None, :], kwargs])
        elif F.ndim == 2:
            self.to_plot.append([F, kwargs])
        elif F.ndim == 3:
            [self.to_plot.append([_F, kwargs.copy()]) for _F in F]

        return self

    def plot_if_not_done_yet(self):
        if self.ax is None:
            self.do()

            legend, kwargs = get_parameter_with_options(self.legend)
            if legend:
                self.ax.legend(**kwargs)

            if self.title:
                title, kwargs = get_parameter_with_options(self.title)
                self.ax.set_title(title, **kwargs)

            if self.tight_layout:
                self.fig.tight_layout()

    def show(self, **kwargs):
        self.plot_if_not_done_yet()

        # in a notebook the plot method need not to be called explicitly
        if not in_notebook():
            self.fig.show(**kwargs)

    def save(self, fname, **kwargs):
        self.plot_if_not_done_yet()
        set_if_none(kwargs, "bbox_inches", "tight")
        self.fig.savefig(fname, **kwargs)
        return self

    def get_labels(self):
        if isinstance(self.labels, list):
            if len(self.labels) != self.n_dim:
                raise Exception("Number of axes labels not equal to the number of axes.")
            else:
                return self.labels
        else:
            return [f"${self.labels}_{{{i}}}$" for i in range(1, self.n_dim + 1)]


def get_parameter_with_options(param):
    if param is None:
        return None, None
    else:
        if isinstance(param, tuple) or isinstance(param, list):
            val, kwargs = param
        else:
            val, kwargs = param, {}

        return val, kwargs
