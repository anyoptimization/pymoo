import sys

import numpy as np
from deprecated import deprecated

from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.replacement import ReplacementSurvival
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.to_bound import (
    ToBoundOutOfBoundsRepair,
    set_to_bounds_if_outside,
)
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from pymoo.util.archive import MultiObjectiveArchive, SurvivalTruncation
from pymoo.util.display.multi import MultiObjectiveOutput


class CrowdingDistanceTournamentSurvival(Survival):
    """
    Tournament selection survival favoring higher crowding distance
    """

    def __init__(self, tournament_size=3):
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


class ParetoDominatedReplacement(ReplacementSurvival):
    def __init__(self):
        super().__init__(filter_infeasible=True)
        self._cd = get_crowding_function("cd")

    def _do(self, problem, pop, off, random_state=None, **kwargs):
        ret = np.full((len(pop), 1), False)
        for i in range(len(off)):
            # Compare new population with current population
            if self._dominates(off[i].F, pop[i].F):
                ret[i] = True
            elif self._dominates(pop[i].F, off[i].F):
                pass  # Keep current population
            else:
                # Non-dominated case: use crowding distance to decide
                # Combine both solutions to calculate crowding distance
                F_combined = np.vstack([pop[i].F, off[i].F])
                try:
                    crowding = self._cd.do(F_combined)
                    # Select the one with higher crowding distance (more diverse)
                    if crowding[1] > crowding[0]:  # new solution has higher crowding
                        ret[i] = True
                    # Otherwise keep current population
                except Exception as e:
                    print(e, file=sys.stderr)
                    # Fallback to random selection if crowding distance fails
                    if random_state.random() < 0.5:
                        ret[i] = True
        return ret[:, 0]

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2"""
        return np.all(f1 <= f2) and np.any(f1 < f2)


def pso_equation(X, P_X, S_X, V, V_max, w, c1, c2, random_state=None):
    rng = np.random.default_rng(random_state)

    r1 = rng.random(X.shape)
    r2 = rng.random(X.shape)

    inerta = w * V
    cognitive = c1 * r1 * (P_X - X)
    social = c2 * r2 * (S_X - X)

    # calculate the velocity vector
    Vp = inerta + cognitive + social
    Vp = set_to_bounds_if_outside(Vp, -V_max, V_max)

    Xp = X + Vp

    return Xp, Vp


@deprecated
class _CMOPSO(Algorithm):
    def __init__(
        self,
        pop_size=100,
        w=0.729844,
        c1=1.49618,
        c2=1.49618,
        max_velocity_rate=0.2,
        archive_size=200,
        initial_velocity="random",  # 'random' | 'zero'
        sampling=FloatRandomSampling(),
        repair=ToBoundOutOfBoundsRepair(),
        termination=None,
        output=MultiObjectiveOutput(),
        display=None,
        callback=None,
        return_least_infeasible=False,
        save_history=False,
        verbose=False,
        seed=None,
        evaluator=None,
        **kwargs,
    ):
        super().__init__(
            termination=termination,
            output=output,
            display=display,
            callback=callback,
            return_least_infeasible=return_least_infeasible,
            save_history=save_history,
            verbose=verbose,
            seed=seed,
            evaluator=evaluator,
            **kwargs,
        )
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.pop_size = pop_size
        self.max_velocity_rate = max_velocity_rate
        self.archive_size = archive_size
        self.initialization = Initialization(sampling)
        self.initial_velocity = initial_velocity
        self.repair = repair
        self.replacement = ParetoDominatedReplacement()
        self._cd = get_crowding_function("cd")

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        self.archive = MultiObjectiveArchive(
            truncation=SurvivalTruncation(
                CrowdingDistanceTournamentSurvival(), problem=problem
            ),
            max_size=self.archive_size,
        )
        self.V_max = self.max_velocity_rate * (problem.xu - problem.xl)
        self.f, self.strategy = None, None

    def _initialize_infill(self):
        return self.initialization.do(
            self.problem, self.pop_size, algorithm=self, random_state=self.random_state
        )

    def _initialize_advance(self, infills=None, **kwargs):
        particles = self.pop

        if self.initial_velocity == "random":
            init_V = (
                self.random_state.random((len(particles), self.problem.n_var))
                * self.V_max[None, :]
            )
        elif self.initial_velocity == "zero":
            init_V = np.zeros((len(particles), self.problem.n_var))
        else:
            raise Exception("Unknown velocity initialization.")

        particles.set("V", init_V)
        self.particles = particles
        self.archive.add(infills)

    def _select_diverse_leaders(self):
        leaders = []
        if len(self.archive) == 0:
            # If no archive, select randomly from population
            for _ in range(self.pop_size):
                if len(self.pop) > 0:
                    idx = self.random_state.integers(0, len(self.pop))
                    leaders.append(self.pop[idx])
                else:
                    leaders.append(None)
            return leaders

        # Ensure each particle gets a potentially different leader
        for _ in range(self.pop_size):
            if len(self.archive) == 1:
                leaders.append(self.archive[0])
            else:
                try:
                    # Use binary tournament selection with crowding distance
                    idx1 = self.random_state.integers(0, len(self.archive))
                    idx2 = self.random_state.integers(0, len(self.archive))
                    if idx1 == idx2:
                        leaders.append(self.archive[idx1])
                    else:
                        # Calculate crowding distance for comparison
                        F = self.archive.get("F")
                        crowding = self._cd.do(F)

                        # Select leader with higher crowding distance (more diverse)
                        if crowding[idx1] > crowding[idx2]:
                            leaders.append(self.archive[idx1])
                        else:
                            leaders.append(self.archive[idx2])
                except Exception as e:
                    print(e, file=sys.stderr)
                    # Fallback to random selection
                    idx = self.random_state.integers(0, len(self.archive))
                    leaders.append(self.archive[idx])
        return np.asarray([leader.X for leader in leaders])

    def _infill(self):
        problem, particles, pbest = self.problem, self.particles, self.pop

        (X, V) = particles.get("X", "V")
        P_X = pbest.get("X")

        # sbest = self._social_best()
        # S_X = sbest.get("X")
        S_X = self._select_diverse_leaders()

        Xp, Vp = pso_equation(
            X,
            P_X,
            S_X,
            V,
            self.V_max,
            self.w,
            self.c1,
            self.c2,
            random_state=self.random_state,
        )

        # create the offspring population
        off = Population.new(X=Xp, V=Vp)
        off = self.repair(problem, off)

        return off

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, (
            "This algorithm uses the AskAndTell interface thus 'infills' must to be provided."
        )

        # set the new population to be equal to the offsprings
        self.particles = infills

        # if an offspring has improved the personal store that index
        has_improved = self.replacement.do(
            self.problem, self.pop, infills, return_indices=True
        )

        combined_pop = Population.merge(self.pop, infills)
        self.archive.add(combined_pop)

        # set the personal best which have been improved
        self.pop[has_improved] = infills[has_improved]

    def _social_best(self):
        return Population([self.opt[0]] * len(self.pop))


def cmopso_equation(X, S_X, V, V_max, random_state=None):
    def unit_vector(vector):
        """Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    rng = np.random.default_rng(random_state)

    W_X = []
    for i in range(np.shape(X)[0]):  # binary tournament selection on archive
        a = S_X[rng.choice(range(np.shape(S_X)[0]))]
        b = S_X[rng.choice(range(np.shape(S_X)[0]))]
        pw = min(a, b, key=lambda x: angle_between(x, X[i]))
        W_X.append(pw)
    W_X = np.asarray(W_X)

    r1 = rng.random(X.shape)
    r2 = rng.random(X.shape)

    # calculate the velocity vector
    Vp = r1 * V + r2 * (W_X - X)
    Vp = set_to_bounds_if_outside(Vp, -V_max, V_max)

    Xp = X + Vp

    return Xp, Vp


class CMOPSO(Algorithm):
    def __init__(
        self,
        pop_size=100,
        w=0.729844,
        c1=1.49618,
        c2=1.49618,
        max_velocity_rate=0.2,
        elite_size=10,
        initial_velocity="random",  # 'random' | 'zero'
        sampling=FloatRandomSampling(),
        repair=ToBoundOutOfBoundsRepair(),
        mutate=False,
        termination=None,
        output=MultiObjectiveOutput(),
        display=None,
        callback=None,
        return_least_infeasible=False,
        save_history=False,
        verbose=False,
        seed=None,
        evaluator=None,
        **kwargs,
    ):
        super().__init__(
            termination=termination,
            output=output,
            display=display,
            callback=callback,
            return_least_infeasible=return_least_infeasible,
            save_history=save_history,
            verbose=verbose,
            seed=seed,
            evaluator=evaluator,
            **kwargs,
        )
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.pop_size = pop_size
        self.max_velocity_rate = max_velocity_rate
        self.elite_size = elite_size
        self.initialization = Initialization(sampling)
        self.initial_velocity = initial_velocity
        self.repair = repair
        self.replacement = ParetoDominatedReplacement()
        self._cd = get_crowding_function("cd")
        self.mutate = mutate

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        self.archive = MultiObjectiveArchive(
            truncation=SurvivalTruncation(
                CrowdingDistanceTournamentSurvival(), problem=problem
            ),
            truncate_size=self.elite_size,
        )
        self.V_max = self.max_velocity_rate * (problem.xu - problem.xl)
        self.f, self.strategy = None, None

    def _initialize_infill(self):
        return self.initialization.do(
            self.problem, self.pop_size, algorithm=self, random_state=self.random_state
        )

    def _initialize_advance(self, infills=None, **kwargs):
        particles = self.pop

        if self.initial_velocity == "random":
            init_V = (
                self.random_state.random((len(particles), self.problem.n_var))
                * self.V_max[None, :]
            )
        elif self.initial_velocity == "zero":
            init_V = np.zeros((len(particles), self.problem.n_var))
        else:
            raise Exception("Unknown velocity initialization.")

        particles.set("V", init_V)
        self.particles = particles
        self.archive.add(infills)

    def _select_diverse_leaders(self):
        leaders = []
        if len(self.archive) == 0:
            # If no archive, select randomly from population
            for _ in range(self.pop_size):
                if len(self.pop) > 0:
                    idx = self.random_state.integers(0, len(self.pop))
                    leaders.append(self.pop[idx])
                else:
                    leaders.append(None)
            return leaders

        # Ensure each particle gets a potentially different leader
        for _ in range(self.pop_size):
            if len(self.archive) == 1:
                leaders.append(self.archive[0])
            else:
                try:
                    # Use binary tournament selection with crowding distance
                    idx1 = self.random_state.integers(0, len(self.archive))
                    idx2 = self.random_state.integers(0, len(self.archive))
                    if idx1 == idx2:
                        leaders.append(self.archive[idx1])
                    else:
                        # Calculate crowding distance for comparison
                        F = self.archive.get("F")
                        crowding = self._cd.do(F)

                        # Select leader with higher crowding distance (more diverse)
                        if crowding[idx1] > crowding[idx2]:
                            leaders.append(self.archive[idx1])
                        else:
                            leaders.append(self.archive[idx2])
                except Exception as e:
                    print(e, file=sys.stderr)
                    # Fallback to random selection
                    idx = self.random_state.integers(0, len(self.archive))
                    leaders.append(self.archive[idx])
        return np.asarray([leader.X for leader in leaders])

    def _infill(self):
        problem, particles, pbest = self.problem, self.particles, self.pop

        (X, V) = particles.get("X", "V")
        P_X = pbest.get("X")

        S_X = self.archive.get("X")

        Xp, Vp = cmopso_equation(
            X,
            S_X,
            V,
            self.V_max,
            random_state=self.random_state,
        )

        # create the offspring population
        off = Population.new(X=Xp, V=Vp)
        if self.mutate:
            mutation = PolynomialMutation(prob=1.0)
            off = mutation(problem, off, random_state=self.random_state)
        off = self.repair(problem, off)

        return off

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, (
            "This algorithm uses the AskAndTell interface thus 'infills' must to be provided."
        )

        # set the new population to be equal to the offsprings
        self.particles = infills

        # if an offspring has improved the personal store that index
        has_improved = self.replacement.do(
            self.problem, self.pop, infills, return_indices=True
        )

        combined_pop = Population.merge(self.pop, infills)
        self.archive.add(combined_pop)

        # set the personal best which have been improved
        self.pop[has_improved] = infills[has_improved]


parse_doc_string(CMOPSO.__init__)
