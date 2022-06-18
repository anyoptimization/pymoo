import matplotlib.pyplot as plt
import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.repair import Repair
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize
from pymoo.problems.single.traveling_salesman import visualize, create_random_tsp_problem
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.visualization.video.callback_video import AnimationCallback

problem = create_random_tsp_problem(50, 100, seed=1)


class StartFromZeroRepair(Repair):

    def _do(self, problem, X, **kwargs):
        I = np.where(X == 0)[1]

        for k in range(len(X)):
            i = I[k]
            X[k] = np.concatenate([X[k, i:], X[k, :i]])

        return X


class PathVisualization(AnimationCallback):

    def notify(self, algorithm):
        if algorithm.n_gen % 10 == 0:

            fig, ax = plt.subplots()

            x = algorithm.opt[0].get("X")
            visualize(problem, x, fig=fig, ax=ax, show=False)
            ax.set_title(f"Generation: {algorithm.n_gen}")
            self.video.record(fig=fig)


algorithm = GA(
    pop_size=20,
    sampling=PermutationRandomSampling(),
    mutation=InversionMutation(),
    crossover=OrderCrossover(),
    repair=StartFromZeroRepair(),
    eliminate_duplicates=True
)

# if the algorithm did not improve the last 200 generations then it will terminate (and disable the max generations)
termination = DefaultSingleObjectiveTermination(period=200, n_max_gen=np.inf)

res = minimize(
    problem,
    algorithm,
    termination,
    # UNCOMMENT to save the visualization
    # callback=PathVisualization(fname="tsp.mp4"),
    verbose=True
)

print(res.F)
print(res.algorithm.evaluator.n_eval)

visualize(problem, res.X)

