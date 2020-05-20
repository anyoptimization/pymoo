import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.optimize import minimize
from pymoo.problems.single.traveling_salesman import TravelingSalesman
from pymoo.operators.crossover.order_crossover import OrderCrossover
from pymoo.operators.mutation.inversion_mutation import InversionMutation
from pymoo.operators.sampling.random_permutation_sampling import RandomPermutationSampling


# visualize a solution
def visualize(problem, x, path=None, label=True):
    with plt.style.context('ggplot'):
        # plot cities using scatter plot
        plt.scatter(problem.cities[:, 0], problem.cities[:, 1], s=250)
        if label:
            # annotate cities
            for i, c in enumerate(problem.cities):
                plt.annotate(str(i), xy=c, fontsize=10, ha="center", va="center", color="white")
        if x is not None:
            # plot route path x
            for i in range(len(x) - 1):
                current = x[i]
                next_ = x[i + 1]
                plt.plot(problem.cities[[current, next_], 0], problem.cities[[current, next_], 1], 'r--')
            # back to the initial city
            end = x[-1]
            start = x[0]
            plt.plot(problem.cities[[end, start], 0], problem.cities[[end, start], 1], 'r--')
            plt.title("Route length: %.4f" % problem.get_route_length(x))
            if path is not None:
                plt.savefig(path)
            plt.show()


# randomly generate 10 cities on [0, 100]
cities = np.random.random((10, 2)) * 100
# create problem
problem = TravelingSalesman(cities)
# solve the problem using genetic algorithm
algorithm = GA(
        pop_size=200,
        eliminate_duplicates=True,
        sampling=RandomPermutationSampling(),
        mutation=InversionMutation(),
        crossover=OrderCrossover()
)
res = minimize(
    problem,
    algorithm,
    termination=('n_eval', 50000),
    verbose=False
)
# visualize the solution
visualize(problem, res.X)

