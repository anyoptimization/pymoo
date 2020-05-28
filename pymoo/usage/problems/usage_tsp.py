import numpy as np

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.model.repair import Repair
from pymoo.operators.crossover.order_crossover import OrderCrossover
from pymoo.operators.mutation.inversion_mutation import InversionMutation
from pymoo.operators.sampling.random_permutation_sampling import PermutationRandomSampling
from pymoo.optimize import minimize
from pymoo.problems.single.traveling_salesman import visualize, create_random_tsp_problem
from pymoo.util.termination.default import SingleObjectiveDefaultTermination

problem = create_random_tsp_problem(100, 100, seed=1)


class StartFromZeroRepair(Repair):

    def _do(self, problem, pop, **kwargs):
        X = pop.get("X")
        I = np.where(X == 0)[1]

        for k in range(len(X)):
            i = I[k]
            x = X[k]
            _x = np.concatenate([x[i:], x[:i]])
            pop[k].set("X", _x)

        return pop


algorithm = GA(
    pop_size=20,
    sampling=PermutationRandomSampling(),
    mutation=InversionMutation(),
    # crossover=EdgeRecombinationCrossover(),
    crossover=OrderCrossover(),
    repair=StartFromZeroRepair(),
    eliminate_duplicates=True
)

# if the algorithm did not improve the last 200 generations then it will terminate (and disable the max generations)
termination = SingleObjectiveDefaultTermination(n_last=200, n_max_gen=np.inf)

res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    verbose=False
)

print(res.F)
print(res.algorithm.evaluator.n_eval)

visualize(problem, res.X)


#
# class PathVisualization(Callback):
#
#     def __init__(self):
#         super().__init__()
#         self.vid = Video(File("tsp.mp4"))
#
#     def notify(self, algorithm):
#         if algorithm.n_gen % 10 == 0:
#             x = algorithm.opt[0].get("X")
#             visualize(problem, x, show=False)
#             plt.title(f"Generation: {algorithm.n_gen}")
#             self.vid.record()
#
#
# algorithm.callback = PathVisualization()
#
# res = minimize(
#     problem,
#     algorithm,
#     termination,
#     seed=1,
#     verbose=False
# )
