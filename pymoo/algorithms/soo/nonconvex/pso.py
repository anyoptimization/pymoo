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
from pymoo.util import default_random_state


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

@default_random_state
def pso_equation(X, P_X, S_X, V, V_max, w, c1, c2, r1=None, r2=None, random_state=None):
    n_particles, n_var = X.shape

    if r1 is None:
        r1 = random_state.random((n_particles, n_var))

    if r2 is None:
        r2 = random_state.random((n_particles, n_var))

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
        return self.initialization.do(self.problem, self.pop_size, algorithm=self, random_state=self.random_state)

    def _initialize_advance(self, infills=None, **kwargs):
        particles = self.pop

        if self.initial_velocity == "random":
            init_V = self.random_state.random((len(particles), self.problem.n_var)) * self.V_max[None, :]
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

        Xp, Vp = pso_equation(X, P_X, S_X, V, self.V_max, self.w, self.c1, self.c2, random_state=self.random_state)

        # if the problem has boundaries to be considered
        if problem.has_bounds():

            for k in range(20):
                # find the individuals which are still infeasible
                m = is_out_of_bounds_by_problem(problem, Xp)

                if len(m) == 0:
                    break

                # actually execute the differential equation
                Xp[m], Vp[m] = pso_equation(X[m], P_X[m], S_X[m], V[m], self.V_max, self.w, self.c1, self.c2, random_state=self.random_state)

            # if still infeasible do a random initialization
            Xp = repair_random_init(Xp, X, *problem.bounds(), random_state=self.random_state)

        # create the offspring population
        off = Population.new(X=Xp, V=Vp)

        # try to improve the current best with a pertubation
        if self.pertube_best:
            k = FitnessSurvival().do(problem, pbest, n_survive=1, return_indices=True)[0]
            mut = PM(prob=0.9, eta=self.random_state.uniform(5, 30), at_least_once=False)
            mutant = mut(problem, Population(Individual(X=pbest[k].X)), random_state=self.random_state)[0]
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

        delta = 0.05 + (self.random_state.random() * 0.05)

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



parse_doc_string(PSO.__init__)
