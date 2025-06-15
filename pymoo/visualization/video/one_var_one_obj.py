from pymoo.visualization.matplotlib import plt
import numpy as np

from pymoo.visualization.video.callback_video import AnimationCallback


class OneVariableOneObjectiveVisualization(AnimationCallback):

    def __init__(self,
                 n_samples_for_surface=10000,
                 **kwargs):
        super().__init__(**kwargs)
        self.last_pop = None
        self.n_samples_for_surface = n_samples_for_surface

    def do(self, problem, algorithm):

        # check whether the visualization can be done or not - throw exception or simply do nothing
        if problem.n_var != 1 or problem.n_obj != 1:
            raise Exception("This visualization can only be used for problems with one variable and one objective!")

        # draw the problem surface
        xl, xu = problem.bounds()
        _X = np.linspace(xl, xu, self.n_samples_for_surface)
        _F = problem.evaluate(_X)
        plt.plot(_X, _F, label="True", color="black", alpha=0.6)
        plt.ylim(xl[0], xu[0])
        plt.ylim(_F.min(), _F.max())

        pop = algorithm.pop

        X, F, CV = pop.get("X", "F", "CV")
        plt.scatter(X[:, 0], F[:, 0], color="blue", marker="o", s=70)

        is_new = np.full(len(pop), True)
        if self.last_pop is not None:
            for k, ind in enumerate(pop):
                if ind in self.last_pop:
                    is_new[k] = False

        # plot the new population
        if is_new.sum() > 0:
            X, F, CV = pop[is_new].get("X", "F", "CV")
            plt.scatter(X[:, 0], F[:, 0], color="red", marker="*", s=70)

        if hasattr(algorithm, "off") and algorithm.off is not None:
            X, F, CV = algorithm.off.get("X", "F", "CV")
            plt.scatter(X[:, 0], F[:, 0], color="purple", marker="*", s=40)

        plt.title(f"Generation: {algorithm.n_gen}")
        plt.legend()

        # store the current population as the last
        self.last_pop = set(pop)



