from pymoo.visualization.fitness_landscape import FitnessLandscape
from pymoo.visualization.video.callback_video import AnimationCallback


class PSOAnimation(AnimationCallback):

    def __init__(self,
                 nth_gen=1,
                 n_samples_for_surface=200,
                 dpi=200,
                 **kwargs):

        super().__init__(nth_gen=nth_gen, dpi=dpi, **kwargs)
        self.n_samples_for_surface = n_samples_for_surface
        self.last_pop = None

    def do(self, problem, algorithm, **kwargs):
        from pymoo.visualization.matplotlib import plt

        if problem.n_var != 2 or problem.n_obj != 1:
            raise Exception(
                "This visualization can only be used for problems with two variables and one objective!")

        # draw the problem surface
        FitnessLandscape(problem,
                         _type="contour",
                         kwargs_contour=dict(alpha=0.3),
                         n_samples=self.n_samples_for_surface,
                         close_on_destroy=False).do()

        # get the population
        off = algorithm.particles
        pop = algorithm.particles if self.last_pop is None else self.last_pop
        pbest = algorithm.pop

        for i in range(len(pop)):
            plt.plot([off[i].X[0], pop[i].X[0]], [off[i].X[1], pop[i].X[1]], color="blue", alpha=0.5)
            plt.plot([pbest[i].X[0], pop[i].X[0]], [pbest[i].X[1], pop[i].X[1]], color="red", alpha=0.5)
            plt.plot([pbest[i].X[0], off[i].X[0]], [pbest[i].X[1], off[i].X[1]], color="red", alpha=0.5)

        X, F, CV = pbest.get("X", "F", "CV")
        plt.scatter(X[:, 0], X[:, 1], edgecolors="red", marker="*", s=70, facecolors='none', label="pbest")

        X, F, CV = off.get("X", "F", "CV")
        plt.scatter(X[:, 0], X[:, 1], color="blue", marker="o", s=30, label="particle")

        X, F, CV = pop.get("X", "F", "CV")
        plt.scatter(X[:, 0], X[:, 1], color="blue", marker="o", s=30, alpha=0.5)

        opt = algorithm.opt
        X, F, CV = opt.get("X", "F", "CV")
        plt.scatter(X[:, 0], X[:, 1], color="black", marker="x", s=100, label="gbest")

        xl, xu = problem.bounds()
        plt.xlim(xl[0], xu[0])
        plt.ylim(xl[1], xu[1])

        plt.title(f"Generation: %s \nf: %.5E" % (algorithm.n_gen, opt[0].F[0]))
        plt.legend()

        self.last_pop = off.copy(deep=True)
