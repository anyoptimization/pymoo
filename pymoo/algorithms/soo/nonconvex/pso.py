import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.algorithm import Algorithm
from pymoo.core.individual import Individual
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair
from pymoo.core.replacement import ImprovementReplacement
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.dex import repair_random_init
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.bounds_repair import is_out_of_bounds_by_problem
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.display.column import Column
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util.misc import norm_eucl_dist
from pymoo.visualization.fitness_landscape import FitnessLandscape
from pymoo.visualization.video.callback_video import AnimationCallback


# =========================================================================================================
# Display
# =========================================================================================================

class PSOFuzzyOutput(SingleObjectiveOutput):

    def __init__(self):
        super().__init__()

        self.f = Column(name="f", width=8)
        self.S = Column(name="S", width=7)
        self.w = Column(name="w", width=7)
        self.c1 = Column(name="c1", width=8)
        self.c2 = Column(name="c2", width=8)

        self.columns += [self.f, self.S, self.w, self.c1, self.c2]

    def update(self, algorithm):
        super().update(algorithm)

        self.f.set(algorithm.f)
        self.S.set(algorithm.strategy)
        self.w.set(algorithm.w)
        self.c1.set(algorithm.c1)
        self.c2.set(algorithm.c2)


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
# Equation
# =========================================================================================================

def pso_equation(X, P_X, S_X, V, V_max, w, c1, c2, r1=None, r2=None):
    n_particles, n_var = X.shape

    if r1 is None:
        r1 = np.random.random((n_particles, n_var))

    if r2 is None:
        r2 = np.random.random((n_particles, n_var))

    inerta = w * V
    cognitive = c1 * r1 * (P_X - X)
    social = c2 * r2 * (S_X - X)

    # calculate the velocity vector
    Vp = inerta + cognitive + social
    Vp = set_to_bounds_if_outside(Vp, - V_max, V_max)

    Xp = X + Vp

    return Xp, Vp


# =========================================================================================================
# Implementation
# =========================================================================================================


class PSO(Algorithm):

    def __init__(self,
                 pop_size=25,
                 sampling=LHS(),
                 w=0.9,
                 c1=2.0,
                 c2=2.0,
                 adaptive=True,
                 initial_velocity="random",
                 max_velocity_rate=0.20,
                 pertube_best=True,
                 repair=NoRepair(),
                 output=PSOFuzzyOutput(),
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
            The inertia F to be used in each iteration for the velocity update. This can be interpreted
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

        super().__init__(output=output, **kwargs)

        self.initialization = Initialization(sampling)

        self.pop_size = pop_size
        self.adaptive = adaptive
        self.pertube_best = pertube_best
        self.V_max = None
        self.initial_velocity = initial_velocity
        self.max_velocity_rate = max_velocity_rate
        self.repair = repair

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.particles = None
        self.sbest = None

    def _setup(self, problem, **kwargs):
        self.V_max = self.max_velocity_rate * (problem.xu - problem.xl)
        self.f, self.strategy = None, None

    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.pop_size, algorithm=self)

    def _initialize_advance(self, infills=None, **kwargs):
        particles = self.pop

        if self.initial_velocity == "random":
            init_V = np.random.random((len(particles), self.problem.n_var)) * self.V_max[None, :]
        elif self.initial_velocity == "zero":
            init_V = np.zeros((len(particles), self.problem.n_var))
        else:
            raise Exception("Unknown velocity initialization.")

        particles.set("V", init_V)
        self.particles = particles

        super()._initialize_advance(infills=infills, **kwargs)

    def _infill(self):
        problem, particles, pbest = self.problem, self.particles, self.pop

        (X, V) = particles.get("X", "V")
        P_X = pbest.get("X")

        sbest = self._social_best()
        S_X = sbest.get("X")

        Xp, Vp = pso_equation(X, P_X, S_X, V, self.V_max, self.w, self.c1, self.c2)

        # if the problem has boundaries to be considered
        if problem.has_bounds():

            for k in range(20):
                # find the individuals which are still infeasible
                m = is_out_of_bounds_by_problem(problem, Xp)

                if len(m) == 0:
                    break

                # actually execute the differential equation
                Xp[m], Vp[m] = pso_equation(X[m], P_X[m], S_X[m], V[m], self.V_max, self.w, self.c1, self.c2)

            # if still infeasible do a random initialization
            Xp = repair_random_init(Xp, X, *problem.bounds())

        # create the offspring population
        off = Population.new(X=Xp, V=Vp)

        # try to improve the current best with a pertubation
        if self.pertube_best:
            k = FitnessSurvival().do(problem, pbest, n_survive=1, return_indices=True)[0]
            mut = PM(prob=0.9, eta=np.random.uniform(5, 30), at_least_once=False)
            mutant = mut(problem, Population(Individual(X=pbest[k].X)))[0]
            off[k].set("X", mutant.X)

        self.repair(problem, off)
        self.sbest = sbest

        return off

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus 'infills' must to be provided."

        # set the new population to be equal to the offsprings
        self.particles = infills

        # if an offspring has improved the personal store that index
        has_improved = ImprovementReplacement().do(self.problem, self.pop, infills, return_indices=True)

        # set the personal best which have been improved
        self.pop[has_improved] = infills[has_improved]

        if self.adaptive:
            self._adapt()

    def _social_best(self):
        return Population([self.opt[0]] * len(self.pop))

    def _adapt(self):
        pop = self.pop

        X, F = pop.get("X", "F")
        sbest = self.sbest
        w, c1, c2, = self.w, self.c1, self.c2

        # get the average distance from one to another for normalization
        D = norm_eucl_dist(self.problem, X, X)
        mD = D.sum(axis=1) / (len(pop) - 1)
        _min, _max = mD.min(), mD.max()

        # get the average distance to the best
        g_D = norm_eucl_dist(self.problem, sbest.get("X"), X).mean()
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


parse_doc_string(PSO.__init__)
