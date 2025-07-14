import numpy as np

from pymoo.algorithms.moo.spea2 import SPEA2Survival
from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.to_bound import (
    ToBoundOutOfBoundsRepair,
    set_to_bounds_if_outside,
)
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from pymoo.util import default_random_state
from pymoo.util.archive import MultiObjectiveArchive, SurvivalTruncation
from pymoo.util.display.multi import MultiObjectiveOutput


class CrowdingDistanceTournamentSurvival(Survival):
    def __init__(self, tournament_size=3):
        """
        Survival strategy that uses tournament selection based on crowding distance.

        This class inherits from :class:`~pymoo.core.survival.Survival` and implements a
        tournament selection mechanism where individuals with higher crowding distance
        (indicating better diversity) are preferred.

        Parameters
        ----------
        tournament_size : int, optional
            The number of individuals participating in each tournament. Defaults to 3.
        """
        super().__init__(filter_infeasible=True)
        self._tournament_size = tournament_size
        self._cd = get_crowding_function("cd")

    def _do(self, problem, pop, n_survive, random_state=None, **kwargs):
        crowding = self._cd.do(pop.get("F"))

        # Select solutions with better crowding distance (more diverse)
        selected_indices = []
        remaining_indices = list(range(len(pop)))

        while len(selected_indices) < n_survive and remaining_indices:
            # Tournament selection favoring higher crowding distance
            tournament_size = min(self._tournament_size, len(remaining_indices))
            tournament_indices = random_state.choice(
                remaining_indices, size=tournament_size, replace=False
            )

            # Select the one with highest crowding distance in tournament
            best_idx = tournament_indices[np.argmax(crowding[tournament_indices])]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return pop[selected_indices]


def angle_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)  # unit vector; +1-e6 to avoid zero-division
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)  # unit vector
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


@default_random_state
def cmopso_equation(X, L, V, V_max, random_state=None):
    """
    Calculates the new positions and velocities of particles based on the CMOPSO
    competition-based learning strategy.

    This function implements the core update equations for the Competitive Mechanism
    based Multi-objective Particle Swarm Optimizer (CMOPSO). Each particle's
    velocity is updated by learning from a "winner" selected from the elite archive
    (`L`) through a binary tournament. The winner is chosen based on the smallest
    angle between the particle's current position and the elite's position, promoting
    diversity.

    Parameters
    ----------
    X : numpy.ndarray
        Current positions of the particles (population). Shape `(n_particles, n_var)`.
    L : numpy.ndarray
        Positions of the elite particles from the archive. Shape `(n_elites, n_var)`.
    V : numpy.ndarray
        Current velocities of the particles. Shape `(n_particles, n_var)`.
    V_max : numpy.ndarray
        Maximum allowed velocity for each decision variable. Shape `(n_var,)`.
    random_state : numpy.random.RandomState, optional
        Random state for reproducibility. If None, a new default random number
        generator is used.

    Returns
    -------
    Xp : numpy.ndarray
        New positions of the particles after the update. Shape `(n_particles, n_var)`.
    Vp : numpy.ndarray
        New velocities of the particles after the update. Shape `(n_particles, n_var)`.
    """

    W_X = []
    for i in range(np.shape(X)[0]):  # binary tournament selection on elites
        aidx = random_state.choice(range(len(L)))
        bidx = random_state.choice(range(len(L)))
        a = L[aidx]
        b = L[bidx]
        pw = min(a, b, key=lambda x: angle_between(x, X[i]))
        W_X.append(pw)
    W_X = np.asarray(W_X)

    r1 = random_state.random(X.shape)
    r2 = random_state.random(X.shape)

    # calculate the velocity vector
    Vp = r1 * V + r2 * (W_X - X)
    Vp = set_to_bounds_if_outside(Vp, -V_max, V_max)

    Xp = X + Vp

    return Xp, Vp


class CMOPSO(Algorithm):
    def __init__(
        self,
        pop_size=100,
        max_velocity_rate=0.2,
        elite_size=10,
        initial_velocity="random",  # 'random' | 'zero'
        mutation_rate=0.5,
        sampling=FloatRandomSampling(),
        repair=ToBoundOutOfBoundsRepair(),
        output=MultiObjectiveOutput(),
        **kwargs,
    ):
        """
        Competitive mechanism based Multi-objective Particle Swarm Optimizer (CMOPSO).

        Particle updates are based on learning from the "winner" of binary tournaments
        of randomly selected elites. Replacement strategy is based on SPEA2.

        Zhang, X., Zheng, X., Cheng, R., Qiu, J., & Jin, Y. (2018). A competitive mechanism
        based multi-objective particle swarm optimizer with fast convergence.
        Inf. Sci., 427, 63-76.

        Parameters
        ----------
        pop_size : int, optional
            The population size. Defaults to 100.
        max_velocity_rate : float, optional
            The maximum velocity rate. Defaults to 0.2.
        max_elite_size : int, optional
            The maximum size of the elite archive. Defaults to 10.
        initial_velocity : str, optional
            Defines how the initial velocity of particles is set. Can be "random" or "zero".
            Defaults to "random".
        mutate_rate : float, optional
            Rate at which to apply polynomial mutation to the offspring. Defaults to 0.5.
        sampling : :class:`~pymoo.core.sampling.Sampling`, optional
            Sampling strategy used to generate the initial population. Defaults to
            :class:`~pymoo.operators.sampling.rnd.FloatRandomSampling`.
        repair : :class:`~pymoo.operators.repair.Repair`, optional
            Repair method for out-of-bounds variables. Defaults to
            :class:`~pymoo.operators.repair.to_bound.ToBoundOutOfBoundsRepair`.
        output : :class:`~pymoo.util.display.output.Output`, optional
            Output object to be used for logging. Defaults to
            :class:`~pymoo.util.display.multi.MultiObjectiveOutput`.
        **kwargs
            Additional keyword arguments to be passed to the Algorithm superclass.
        """
        super().__init__(output=output, **kwargs)
        self.pop_size = pop_size
        self.max_velocity_rate = max_velocity_rate
        self.elite_size = elite_size
        self.initialization = Initialization(sampling)
        self.initial_velocity = initial_velocity
        self.repair = repair
        self._cd = get_crowding_function("cd")
        self.mutation = PolynomialMutation(prob=mutation_rate)
        self.survival = SPEA2Survival()

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        self.elites = MultiObjectiveArchive(
            truncation=SurvivalTruncation(
                CrowdingDistanceTournamentSurvival(), problem=problem
            ),
            max_size=self.pop_size,
            truncate_size=self.elite_size,
        )
        self.V_max = self.max_velocity_rate * (problem.xu - problem.xl)

    def _initialize_infill(self):
        return self.initialization.do(
            self.problem, self.pop_size, algorithm=self, random_state=self.random_state
        )

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills

        if self.initial_velocity == "random":
            init_V = (
                self.random_state.random((len(self.pop), self.problem.n_var))
                * self.V_max[None, :]
            )
        elif self.initial_velocity == "zero":
            init_V = np.zeros((len(self.pop), self.problem.n_var))
        else:
            raise Exception("Unknown velocity initialization.")

        self.pop.set("V", init_V)
        self.elites = self.elites.add(self.pop)

    def _infill(self):
        (X, V) = self.pop.get("X", "V")
        L = self.elites.get("X")

        Xp, Vp = cmopso_equation(X, L, V, self.V_max, random_state=self.random_state)

        # create the offspring population
        off = Population.new(X=Xp, V=Vp)
        off = self.mutation(self.problem, off, random_state=self.random_state)
        off = self.repair(self.problem, off)

        return off

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, (
            "This algorithm uses the AskAndTell interface thus 'infills' must to be provided."
        )

        particles = Population.merge(self.pop, infills)
        self.elites = self.elites.add(particles)
        self.pop = self.survival.do(self.problem, particles, n_survive=self.pop_size)


parse_doc_string(CrowdingDistanceTournamentSurvival.__init__)
parse_doc_string(CMOPSO.__init__)
