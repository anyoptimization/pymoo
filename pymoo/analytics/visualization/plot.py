import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pymoo.operators.default_operators import set_if_none
from pymoo.util.normalization import normalize


class Plot():

    def __init__(self,
                 figsize=(8, 6),
                 title=None,
                 legend=False,
                 tight_layout=False,
                 bounds=None,
                 cmap=matplotlib.cm.Dark2.colors,
                 labels="f"):

        super().__init__()
        self.fig = plt.figure(figsize=figsize)
        self.cmap = cmap
        self.to_plot = []
        self.ax = None
        self.legend = legend
        self.tight_layout = tight_layout

        self.labels = labels
        self.title = title
        self.n_dim = None

        self.bounds = bounds

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
        for k in range(len(self.to_plot)):
            self.to_plot[k][0] = normalize(self.to_plot[k][0], self.bounds[0], self.bounds[1])

    def apply(self, func):
        func(self.ax)
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

            if self.legend:
                self.ax.legend()

            if self.title:
                self.ax.title(self.title)

            if self.tight_layout:
                self.fig.tight_layout()

    def show(self, **kwargs):
        self.plot_if_not_done_yet()
        self.fig.show(**kwargs)
        return self

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


def number_to_text(val):
    if val > 1e3:
        return "{:.2e}".format(val)
    else:
        return "{:.2f}".format(val)
