import importlib

import numpy as np
from pymoo.visualization.matplotlib import matplotlib, plt, colors, ListedColormap

from pymoo.util.misc import set_if_none
from pymoo.visualization.util import default_number_to_text, in_notebook


class Plot:

    def __init__(self,
                 fig=None,
                 ax=None,
                 figsize=(8, 6),
                 title=None,
                 legend=False,
                 tight_layout=False,
                 bounds=None,
                 reverse=False,
                 cmap="tab10",
                 axis_style=None,
                 axis_label_style=None,
                 func_number_to_text=default_number_to_text,
                 labels="f",
                 close_on_destroy=True,
                 **kwargs):

        super().__init__()

        # change the font of plots to serif (looks better)
        plt.rc('font', family='serif')

        # the matplotlib classes
        self.fig = fig
        self.ax = ax
        self.figsize = figsize

        # whether the figure should be closed when object is destroyed
        self.close_on_destroy = close_on_destroy

        # the title of the figure - can also be a list if subfigures
        self.title = title

        # the style to be used for the axis
        if axis_style is None:
            self.axis_style = {}
        else:
            self.axis_style = axis_style.copy()

        # the style of the axes
        if axis_label_style is None:
            self.axis_label_style = {}
        else:
            self.axis_label_style = axis_label_style.copy()

        # how numbers are represented if plotted
        self.func_number_to_text = func_number_to_text

        # if data are normalized reverse can be applied
        self.reverse = reverse

        # the labels for each axis
        self.axis_labels = labels

        # the data to plot
        self.to_plot = []

        # whether to plot a legend or apply tight layout
        self.legend = legend
        self.tight_layout = tight_layout

        # the colormap or the color lists to use
        if isinstance(cmap, str):
            self.cmap = matplotlib.pyplot.get_cmap(cmap)
        else:
            self.cmap = cmap
        if isinstance(self.cmap, ListedColormap):
            self.colors = self.cmap.colors

        # the dimensionality of the data
        self.n_dim = None

        # the boundaries for normalization
        self.bounds = bounds

    def init_figure(self, n_rows=1, n_cols=1, plot_3D=False, force_axes_as_matrix=False):
        if self.ax is not None:
            return

        if not plot_3D:
            self.fig, self.ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=self.figsize)
        else:
            importlib.import_module("mpl_toolkits.mplot3d")
            self.fig = plt.figure(figsize=self.figsize)
            self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')

        # if there is more than one figure we represent it as a 2D numpy array
        if (n_rows > 1 or n_cols > 1) or force_axes_as_matrix:
            self.ax = np.array(self.ax).reshape(n_rows, n_cols)

    def do(self):

        if len(self.to_plot) > 0:
            unique_dim = np.unique(np.array([e[0].shape[-1] for e in self.to_plot]))
            if len(unique_dim) > 1:
                raise Exception("Inputs with different dimensions were added: %s" % unique_dim)

            self.n_dim = unique_dim[0]

        # actually call the class
        self._do()

        # convert the axes to a list
        axes = np.array(self.ax).flatten()

        for i, ax in enumerate(axes):

            legend, kwargs = get_parameter_with_options(self.legend)
            if legend:
                ax.legend(**kwargs)

            title, kwargs = get_parameter_with_options(self.title)
            if self.title:
                if isinstance(self.title, list):
                    ax.set_title(title[i], **kwargs)
                else:
                    ax.set_title(title, **kwargs)

        if self.tight_layout:
            self.fig.tight_layout()

        return self

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

        if F is None:
            return self
        elif F.ndim == 1:
            self.to_plot.append([F[None, :], kwargs])
        elif F.ndim == 2:
            self.to_plot.append([F, kwargs])
        elif F.ndim == 3:
            [self.to_plot.append([_F, kwargs.copy()]) for _F in F]

        return self

    def plot_if_not_done_yet(self):
        if self.ax is None:
            self.do()

    def show(self, **kwargs):
        self.plot_if_not_done_yet()

        # in a notebook the plot method need not to be called explicitly
        if not in_notebook() and matplotlib.get_backend() != "agg":
            plt.show(**kwargs)
            plt.close()

        return self

    def save(self, fname, **kwargs):
        self.plot_if_not_done_yet()
        set_if_none(kwargs, "bbox_inches", "tight")
        self.fig.savefig(fname, **kwargs)
        return self

    def get_labels(self):
        if isinstance(self.axis_labels, list):
            if len(self.axis_labels) != self.n_dim:
                raise Exception("Number of axes labels not equal to the number of axes.")
            else:
                return self.axis_labels
        else:
            return [f"${self.axis_labels}_{{{i}}}$" for i in range(1, self.n_dim + 1)]

    def __del__(self):
        if self.fig is not None and self.close_on_destroy:
            plt.close(self.fig)


def get_parameter_with_options(param):
    if param is None:
        return None, None
    else:
        if isinstance(param, tuple):
            val, kwargs = param
        else:
            val, kwargs = param, {}

        return val, kwargs
