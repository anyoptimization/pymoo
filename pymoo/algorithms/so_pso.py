import numpy as np

from pymoo.algorithms.so_genetic_algorithm import FitnessSurvival
from pymoo.docs import parse_doc_string
from pymoo.model.algorithm import Algorithm, filter_optimum
from pymoo.model.initialization import Initialization
from pymoo.model.population import Population
from pymoo.model.replacement import ImprovementReplacement, is_better
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.repair.inverse_penalty import InversePenaltyOutOfBoundsRepair
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.misc import norm_eucl_dist, norm_euclidean_distance
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from pymoo.visualization.fitness_landscape import FitnessLandscape
from pymoo.visualization.video.callback_video import AnimationCallback


# =========================================================================================================
# Display
# =========================================================================================================

class PSODisplay(SingleObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        pop = algorithm.pop
        algorithm.pop = Population.create(*algorithm.pop.get("pbest"))
        super()._do(problem, evaluator, algorithm)
        algorithm.pop = pop

        if algorithm.adaptive:
            self.output.append("f", algorithm.f if algorithm.f is not None else "-", width=8)
            self.output.append("S", algorithm.strategy if algorithm.strategy is not None else "-", width=6)
            self.output.append("w", algorithm.w, width=6)
            self.output.append("c1", algorithm.c1, width=8)
            self.output.append("c2", algorithm.c2, width=8)


# =========================================================================================================
# Adaptation Constants
# =========================================================================================================


def S1_exploration(f):
    if f <= 0.4:
        return 0
    elif 0.4 < f <= 0.6:
        return 5 * f - 2
    elif 0.6 < f <= 0.7:
        return 1
    elif 0.7 < f <= 0.8:
        return -10 * f + 8
    elif 0.8 < f:
        return 0


def S2_exploitation(f):
    if f <= 0.2:
        return 0
    elif 0.2 < f <= 0.3:
        return 10 * f - 2
    elif 0.3 < f <= 0.4:
        return 1
    elif 0.4 < f <= 0.6:
        return -5 * f + 3
    elif 0.6 < f:
        return 0


def S3_convergence(f):
    if f <= 0.1:
        return 1
    elif 0.1 < f <= 0.3:
        return -5 * f + 1.5
    elif 0.3 < f:
        return 0


def S4_jumping_out(f):
    if f <= 0.7:
        return 0
    elif 0.7 < f <= 0.9:
        return 5 * f - 3.5
    elif 0.9 < f:
        return 1


# =========================================================================================================
# Implementation
# =========================================================================================================


class PSO(Algorithm):

    def __init__(self,
                 pop_size=25,
                 sampling=LatinHypercubeSampling(),
                 w=0.9,
                 c1=2.0,
                 c2=2.0,
                 adaptive=True,
                 initial_velocity="random",
                 max_velocity_rate=0.20,
                 pertube_best=True,
                 display=PSODisplay(),
                 **kwargs):
        """

        Parameters
        ----------
        pop_size : The size of the swarm being used.

        sampling : {sampling}

        adaptive : bool
            Whether w, c1, and c2 are changed dynamically over time. The update uses the spread from the global
            optimum to determine suitable values.

        w : float
            The inertia weight to be used in each iteration for the velocity update. This can be interpreted
            as the momentum term regarding the velocity. If `adaptive=True` this is only the
            initially used value.

        c1 : float
            The cognitive impact (personal best) during the velocity update. If `adaptive=True` this is only the
            initially used value.
        c2 : float
            The social impact (global best) during the velocity update. If `adaptive=True` this is only the
            initially used value.

        initial_velocity : str - ('random', or 'zero')
            How the initial velocity of each particle should be assigned. Either 'random' which creates a
            random velocity vector or 'zero' which makes the particles start to find the direction through the
            velocity update equation.


        max_velocity_rate : float
            The maximum velocity rate. It is determined variable (and not vector) wise. We consider the rate here
            since the value is normalized regarding the `xl` and `xu` defined in the problem.

        pertube_best : bool
            Some studies have proposed to mutate the global best because it has been found to converge better.
            Which means the population size is reduced by one particle and one function evaluation is spend
            additionally to permute the best found solution so far.

        """

        super().__init__(display=display, **kwargs)

        self.initialization = Initialization(sampling)

        self.pop_size = pop_size
        self.adaptive = adaptive
        self.pertube_best = pertube_best
        self.default_termination = SingleObjectiveDefaultTermination()
        self.V_max = None
        self.initial_velocity = initial_velocity
        self.max_velocity_rate = max_velocity_rate

        self.w = w
        self.c1 = c1
        self.c2 = c2

    def setup(self, problem, **kwargs):
        super().setup(problem, **kwargs)
        self.V_max = self.max_velocity_rate * (problem.xu - problem.xl)

    def _initialize(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        self.evaluator.eval(self.problem, pop, algorithm=self)

        if self.pertube_best:
            pop = FitnessSurvival().do(self.problem, pop, self.pop_size - 1)

        if self.initial_velocity == "random":
            init_V = np.random.random((len(pop), self.problem.n_var)) * self.V_max[None, :]
        elif self.initial_velocity == "zero":
            init_V = np.zeros((len(pop), self.problem.n_var))

        pop.set("V", init_V)
        pop.set("pbest", pop)
        self.pop, self.off = pop, pop

        self.f = None
        self.strategy = None

    def _next(self):
        self._step()

        if self.adaptive:
            self._adapt()

    def _step(self):
        pop = self.pop
        X, F, V = pop.get("X", "F", "V")

        # get the personal best of each particle
        pbest = Population.create(*pop.get("pbest"))
        P_X, P_F = pbest.get("X", "F")

        # get the GLOBAL best solution - other variants such as local best can be implemented here too
        best = self.opt.repeat(len(pop))
        G_X = best.get("X")

        # get the inertia weight of the individual
        inerta = self.w * V

        # calculate random values for the updates
        r1 = np.random.random((len(pop), self.problem.n_var))
        r2 = np.random.random((len(pop), self.problem.n_var))

        cognitive = self.c1 * r1 * (P_X - X)
        social = self.c2 * r2 * (G_X - X)

        # calculate the velocity vector
        _V = inerta + cognitive + social
        _V = set_to_bounds_if_outside(_V, - self.V_max, self.V_max)

        # update the values of each particle
        _X = X + _V
        _X = InversePenaltyOutOfBoundsRepair().do(self.problem, _X, P=X)

        # evaluate the offspring population
        off = Population(len(pop)).set("X", _X, "V", _V, "pbest", pbest)
        self.evaluator.eval(self.problem, off, algorithm=self)

        # check whether a solution has improved or not - also consider constraints here
        has_improved = ImprovementReplacement().do(self.problem, pbest, off, return_indices=True)

        # replace the personal best of each particle if it has improved
        off[has_improved].set("pbest", off[has_improved])
        off.set("best", best)
        pop = off

        # try to improve the current best with a pertubation
        if self.pertube_best:
            pbest = Population.create(*pop.get("pbest"))
            k = FitnessSurvival().do(self.problem, pbest, 1, return_indices=True)[0]
            eta = int(np.random.uniform(5, 30))
            mutant = PolynomialMutation(eta).do(self.problem, pbest[[k]])[0]
            self.evaluator.eval(self.problem, mutant, algorithm=self)

            # if the mutant is in fact better - replace the personal best
            if is_better(mutant, pop[k]):
                pop[k].set("pbest", mutant)

        self.pop = pop

    def _adapt(self):
        pop = self.pop

        X, F, best = pop.get("X", "F", "best")
        best = Population.create(*best)
        w, c1, c2, = self.w, self.c1, self.c2

        # get the average distance from one to another for normalization
        D = norm_eucl_dist(self.problem, X, X)
        mD = D.sum(axis=1) / (len(pop) - 1)
        _min, _max = mD.min(), mD.max()

        # get the average distance to the global best
        g_D = norm_euclidean_distance(self.problem)(best.get("X"), X).mean()
        f = (g_D - _min) / (_max - _min + 1e-32)

        S = np.array([S1_exploration(f), S2_exploitation(f), S3_convergence(f), S4_jumping_out(f)])
        strategy = S.argmax() + 1

        delta = 0.05 + (np.random.random() * 0.05)

        if strategy == 1:
            c1 += delta
            c2 -= delta
        elif strategy == 2:
            c1 += 0.5 * delta
            c2 -= 0.5 * delta
        elif strategy == 3:
            c1 += 0.5 * delta
            c2 += 0.5 * delta
        elif strategy == 4:
            c1 -= delta
            c2 += delta

        c1 = max(1.5, min(2.5, c1))
        c2 = max(1.5, min(2.5, c2))

        if c1 + c2 > 4.0:
            c1 = 4.0 * (c1 / (c1 + c2))
            c2 = 4.0 * (c2 / (c1 + c2))

        w = 1 / (1 + 1.5 * np.exp(-2.6 * f))

        self.f = f
        self.strategy = strategy
        self.c1 = c1
        self.c2 = c2
        self.w = w

    def _set_optimum(self, force=False):
        pbest = Population.create(*self.pop.get("pbest"))
        self.opt = filter_optimum(pbest, least_infeasible=True)


# =========================================================================================================
# Animation
# =========================================================================================================

class PSOAnimation(AnimationCallback):

    def __init__(self,
                 nth_gen=1,
                 n_samples_for_surface=200,
                 dpi=200,
                 **kwargs):
        super().__init__(nth_gen=nth_gen, dpi=dpi, **kwargs)
        self.n_samples_for_surface = n_samples_for_surface
        self.last_pop = None

    def do(self, problem, algorithm):
        import matplotlib.pyplot as plt

        if problem.n_var != 2 or problem.n_obj != 1:
            raise Exception(
                "This visualization can only be used for problems with two variables and one objective!")

        # draw the problem surface
        FitnessLandscape(problem,
                         _type="contour",
                         kwargs_contour=dict(alpha=0.3),
                         n_samples=self.n_samples_for_surface).do()

        # get the population
        off = algorithm.pop
        pop = algorithm.pop if self.last_pop is None else self.last_pop
        pbest = Population.create(*off.get("pbest"))

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


parse_doc_string(PSO.__init__)
