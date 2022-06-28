import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from pymoo.core.problem import ElementwiseProblem


class TravelingSalesman(ElementwiseProblem):

    def __init__(self, cities, **kwargs):
        """
        A two-dimensional traveling salesman problem (TSP)

        Parameters
        ----------
        cities : numpy.array
            The cities with 2-dimensional coordinates provided by a matrix where where city is represented by a row.

        """
        n_cities, _ = cities.shape

        self.cities = cities
        self.D = cdist(cities, cities)

        super(TravelingSalesman, self).__init__(
            n_var=n_cities,
            n_obj=1,
            xl=0,
            xu=n_cities,
            vtype=int,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = self.get_route_length(x)

    def get_route_length(self, x):
        n_cities = len(x)
        dist = 0
        for k in range(n_cities - 1):
            i, j = x[k], x[k + 1]
            dist += self.D[i, j]

        last, first = x[-1], x[0]
        dist += self.D[last, first]  # back to the initial city
        return dist


def create_random_tsp_problem(n_cities, grid_width=100.0, grid_height=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    grid_height = grid_height if grid_height is not None else grid_width
    cities = np.random.random((n_cities, 2)) * [grid_width, grid_height]
    return TravelingSalesman(cities)


def visualize(problem, x, fig=None, ax=None, show=True, label=True):
    with plt.style.context('ggplot'):

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        # plot cities using scatter plot
        ax.scatter(problem.cities[:, 0], problem.cities[:, 1], s=250)
        if label:
            # annotate cities
            for i, c in enumerate(problem.cities):
                ax.annotate(str(i), xy=c, fontsize=10, ha="center", va="center", color="white")

        # plot the line on the path
        for i in range(len(x)):
            current = x[i]
            next_ = x[(i + 1) % len(x)]
            ax.plot(problem.cities[[current, next_], 0], problem.cities[[current, next_], 1], 'r--')

        fig.suptitle("Route length: %.4f" % problem.get_route_length(x))

        if show:
            fig.show()
