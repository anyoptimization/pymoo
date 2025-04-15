import matplotlib.pyplot as plt
import numpy as np

from pymoo.visualization.fitness_landscape import FitnessLandscape
from pymoo.visualization.video.callback_video import AnimationCallback


class TwoVariablesOneObjectiveVisualization(AnimationCallback):

    def __init__(self,
                 n_samples_for_surface=10000,
                 **kwargs):
        super().__init__(**kwargs)
        self.last_pop = None
        self.n_samples_for_surface = n_samples_for_surface

    def do(self, problem, algorithm):

        # check whether the visualization can be done or not - throw exception or simply do nothing
        if problem.n_var != 2 or problem.n_obj != 1:
            raise Exception("This visualization can only be used for problems with two variables and one objective!")

        # draw the problem surface
        # if algorithm.surrogate.targets["F"].doe is not None:
        #     problem = algorithm.surrogate
        plot = FitnessLandscape(problem, _type="contour", kwargs_contour=dict(alpha=0.5))
        plot.do()

        # get the population
        pop = algorithm.pop

        X, F, CV = pop.get("X", "F", "CV")
        plt.scatter(X[:, 0], X[:, 1], facecolor="none", edgecolors="black", marker="o", s=50, label="Solutions")

        if hasattr(algorithm, "off") and algorithm.off is not None:
            X, F, CV = algorithm.off.get("X", "F", "CV")
            plt.scatter(X[:, 0], X[:, 1], color="green", marker="D", s=30, label="Offsprings")

        is_new = np.full(len(pop), True)
        if self.last_pop is not None:
            for k, ind in enumerate(pop):
                if ind in self.last_pop:
                    is_new[k] = False

        # plot the new population
        if is_new.sum() > 0:
            X, F, CV = pop[is_new].get("X", "F", "CV")
            plt.scatter(X[:, 0], X[:, 1], color="red", marker="*", s=70, label="Survivors")

        xl, xu = problem.bounds()
        plt.xlim(xl[0], xu[0])
        plt.ylim(xl[1], xu[1])

        plt.title(f"Generation: {algorithm.n_gen}")
        plt.legend()

        # store the current population as the last
        self.last_pop = set(pop)

        plt.show()

        return plt.gcf()
