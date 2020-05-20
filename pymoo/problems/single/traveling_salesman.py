from scipy.spatial.distance import pdist, squareform
import numpy as np

from pymoo.model.problem import Problem


class TravelingSalesman(Problem):
    """
    2-dimensional travelling salesman problem. This problem uses permutation encoding.
    Args:
        cities: a n by 2 numpy array, where n is the number of cities
    """
    def __init__(self, cities, **kwargs):
        n_cities = cities.shape[0]

        self.cities = cities
        self.PD = squareform(pdist(cities))  # pairwise distance between cities

        super(TravelingSalesman, self).__init__(
            n_var=n_cities, n_obj=1, xl=0, xu=n_cities, type_var=np.int,
            elementwise_evaluation=True,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = self.get_route_length(x)

    def get_route_length(self, x):
        n_cities = len(x)
        dist = 0
        for j in range(n_cities - 1):
            dist = dist + self.PD[x[j], x[j + 1]]
        dist = dist + self.PD[x[-1], x[0]]  # back to the initial city
        return dist