import numpy as np

import pymoo.visualization.util
from pymoo.docs import parse_doc_string
from pymoo.core.plot import Plot
from pymoo.util.misc import all_combinations


class FitnessLandscape(Plot):

    def __init__(self,
                 problem,
                 _type="surface+contour",
                 n_samples=100,
                 colorbar=False,
                 contour_levels=30,
                 kwargs_surface=None,
                 kwargs_contour=None,
                 kwargs_contour_labels=None,
                 **kwargs):

        """

        Fitness Landscape

        Parameters
        ----------------

        problem : The problem to be plotted
        _type : str
            Either "contour", "surface" or "contour+surface"
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
        FitnessLandscape : :class:`~pymoo.core.analytics.visualization.fitness_landscape.FitnessLandscape`

        """

        super().__init__(**kwargs)
        self.problem = problem
        self.n_samples = n_samples
        self._type = _type
        self.colorbar = colorbar

        self.contour_levels = contour_levels

        self.kwargs_surface = kwargs_surface
        if self.kwargs_surface is None:
            self.kwargs_surface = dict(cmap="summer", rstride=1, cstride=1)

        self.kwargs_contour = kwargs_contour
        if self.kwargs_contour is None:
            self.kwargs_contour = dict(linestyles="solid", offset=-1)

        self.kwargs_contour_labels = kwargs_contour_labels

    def _do(self):

        problem, n_samples, _type = self.problem, self.n_samples, self._type

        if problem.n_var == 1 and problem.n_obj == 1:

            self.init_figure()

            X = np.linspace(problem.xl[0], problem.xu[0], num=n_samples)[:, None]
            Z = problem.evaluate(X, return_values_of=["F"])
            pymoo.visualization.util.plot(X, Z)
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("f(x)")

        elif problem.n_var == 2 and problem.n_obj == 1:

            A = np.linspace(problem.xl[0], problem.xu[0], n_samples)
            B = np.linspace(problem.xl[1], problem.xu[1], n_samples)
            X = all_combinations(A, B)

            F = np.reshape(problem.evaluate(X, return_values_of=["F"]), (n_samples, n_samples))

            _X = X[:, 0].reshape((n_samples, n_samples))
            _Y = X[:, 1].reshape((n_samples, n_samples))
            _Z = F.reshape((n_samples, n_samples))

            def plot_surface():
                surf = self.ax.plot_surface(_X, _Y, _Z, **self.kwargs_surface)

                if self.colorbar:
                    self.fig.colorbar(surf)

            def plot_contour():
                CS = self.ax.contour(_X, _Y, _Z, self.contour_levels, **self.kwargs_contour)
                if self.kwargs_contour_labels is not None:
                    self.ax.clabel(CS, **self.kwargs_contour_labels)

                if self.colorbar:
                    self.fig.colorbar(CS)

            if _type == "surface":
                self.init_figure(plot_3D=True)
                plot_surface()
            elif _type == "contour":
                self.init_figure(plot_3D=False)

                if "offset" in self.kwargs_contour:
                    del self.kwargs_contour["offset"]

                plot_contour()
            elif _type == "surface+contour":

                self.init_figure(plot_3D=True)
                plot_surface()
                plot_contour()

        else:
            raise Exception("Only landscapes of problems with one or two variables and one objective can be visualized.")


parse_doc_string(FitnessLandscape.__init__)
