import numpy as np

from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.docs import parse_doc_string
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util import default_random_state
from pymoo.util.archive import MultiObjectiveArchive
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class MOPSO_CD(Algorithm):
    """
    Multi-Objective Particle Swarm Optimization with Crowding Distance (MOPSO-CD) algorithm.

    This implementation extends MOPSO with a crowding distance mechanism for leader selection
    and archive management to ensure a well-distributed Pareto front, suitable for problems
    like MO-HalfCheetah in multi-objective reinforcement learning.

    Parameters
    ----------
    pop_size : int
        The population size (number of particles)
    w : float
        Inertia weight
    c1 : float
        Cognitive parameter (personal best influence)
    c2 : float
        Social parameter (global best influence)
    max_velocity_rate : float
        Maximum velocity rate relative to the variable range
    archive_size : int
        Maximum size of the external archive
    sampling : Sampling
        Sampling strategy for initialization
    output : Output
        Output display
    """

    def __init__(
        self,
        pop_size=100,
        w=0.6,  # Increased for better exploration
        c1=2.0,
        c2=2.0,
        max_velocity_rate=0.5,  # Increased for better exploration
        archive_size=200,  # Increased for better diversity
        sampling=FloatRandomSampling(),
        output=MultiObjectiveOutput(),
        **kwargs,
    ):
        super().__init__(output=output, **kwargs)

        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_velocity_rate = max_velocity_rate
        self.archive_size = archive_size
        self.sampling = sampling

        # Initialize termination if not provided
        self.termination = DefaultMultiObjectiveTermination()

        # Initialize components
        self.nds = NonDominatedSorting()
        self.cd = get_crowding_function("cd")
        self.archive = None
        self.velocities = None
        self.pbest = None
        self.pbest_f = None
        self.random_state = default_random_state(kwargs.get("seed"))

    def _setup(self, problem, **kwargs):
        """Setup the algorithm for the given problem"""
        super()._setup(problem, **kwargs)

        # Initialize the external archive
        self.archive = MultiObjectiveArchive(max_size=self.archive_size)

        # Compute maximum velocity based on problem bounds
        xl, xu = problem.bounds()
        self.v_max = self.max_velocity_rate * (xu - xl)

        # Initialize particles, velocities, and personal bests
        self.pop = self.sampling.do(
            problem, self.pop_size, random_state=self.random_state
        )
        self.velocities = self.random_state.uniform(
            -self.v_max, self.v_max, (self.pop_size, problem.n_var)
        )
        self.pbest = self.pop.copy()  # Personal bests
        self.pbest_f = np.full(
            (self.pop_size, problem.n_obj), np.inf
        )  # Initialize with inf

        # Evaluate initial population to set personal best objectives
        self.evaluator.eval(self.problem, self.pop)

    def _initialize_infill(self):
        """Initialize the population and velocities"""
        # Initialize population using sampling
        pop = self.sampling.do(self.problem, self.pop_size, random_state=self.random_state)

        # Initialize velocities randomly
        self.velocities = self.random_state.uniform(
            -self.v_max, self.v_max, size=(self.pop_size, self.problem.n_var)
        )

        # Initialize personal best (initially same as current positions)
        self.pbest = pop.copy()

        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        """Initialize after evaluation"""
        self.pop = infills

        # Update archive with initial population
        self.archive = self._update_archive(infills)

        # Initialize personal best fitness
        self.pbest = infills.copy()
        self.pbest_f = infills.get("F").copy()

    def _infill(self):
        """Generate new solutions using PSO operators"""
        # Create new population
        X_new = np.zeros((self.pop_size, self.problem.n_var))

        # Pre-select leaders for all particles to ensure diversity
        leaders = self._select_diverse_leaders()

        for i in range(self.pop_size):
            # Use pre-selected leader for this particle
            leader = leaders[i] if leaders[i] is not None else self.pop[i]

            # Generate random coefficients
            r1 = self.random_state.random(self.problem.n_var)
            r2 = self.random_state.random(self.problem.n_var)

            # Update velocity
            cognitive = self.c1 * r1 * (self.pbest[i].X - self.pop[i].X)
            social = self.c2 * r2 * (leader.X - self.pop[i].X)

            self.velocities[i] = self.w * self.velocities[i] + cognitive + social

            # Apply velocity bounds
            self.velocities[i] = set_to_bounds_if_outside(
                self.velocities[i], -self.v_max, self.v_max
            )

            # Update position
            X_new[i] = self.pop[i].X + self.velocities[i]

        # Apply bounds to positions
        xl, xu = self.problem.bounds()
        X_new = set_to_bounds_if_outside(X_new, xl, xu)

        return Population.new("X", X_new)

    def _advance(self, infills=None, **kwargs):
        """Advance the algorithm state"""
        if infills is None:
            return

        # Update archive with crowding distance-based pruning
        combined_pop = Population.merge(self.pop, infills)
        self.archive = self._update_archive(combined_pop)

        # Update personal best
        self._update_pbest(infills)

        # Update current population
        self.pop = infills

    def _update_archive(self, pop):
        """Update the external archive with non-dominated solutions using crowding distance"""
        if len(pop) == 0:
            return self.archive

        # Combine current archive with new solutions
        if len(self.archive) > 0:
            combined = Population.merge(self.archive, pop)
        else:
            combined = pop

        # Find non-dominated solutions
        F = combined.get("F")
        I = self.nds.do(F, only_non_dominated_front=True)
        non_dominated = combined[I]

        # Apply archive size limit using crowding distance
        if len(non_dominated) > self.archive_size:
            # Use tournament selection to maintain diversity while keeping quality
            crowding = self.cd.do(non_dominated.get("F"))

            # Select solutions with better crowding distance (more diverse)
            selected_indices = []
            remaining_indices = list(range(len(non_dominated)))

            while len(selected_indices) < self.archive_size and remaining_indices:
                # Tournament selection favoring higher crowding distance
                tournament_size = min(3, len(remaining_indices))
                tournament_indices = self.random_state.choice(
                    remaining_indices, size=tournament_size, replace=False
                )

                # Select the one with highest crowding distance in tournament
                best_idx = tournament_indices[np.argmax(crowding[tournament_indices])]
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

            non_dominated = non_dominated[selected_indices]

        # Create new archive
        return MultiObjectiveArchive(
            individuals=non_dominated, max_size=self.archive_size
        )

    def _select_diverse_leaders(self):
        """Select diverse leaders for all particles"""
        leaders = []

        if len(self.archive) == 0:
            # If no archive, select randomly from population
            for i in range(self.pop_size):
                if len(self.pop) > 0:
                    idx = self.random_state.integers(0, len(self.pop))
                    leaders.append(self.pop[idx])
                else:
                    leaders.append(None)
            return leaders

        # Ensure each particle gets a potentially different leader
        for i in range(self.pop_size):
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
                        crowding = self.cd.do(F)

                        # Select leader with higher crowding distance (more diverse)
                        if crowding[idx1] > crowding[idx2]:
                            leaders.append(self.archive[idx1])
                        else:
                            leaders.append(self.archive[idx2])
                except Exception:
                    # Fallback to random selection
                    idx = self.random_state.integers(0, len(self.archive))
                    leaders.append(self.archive[idx])

        return leaders

    def _update_pbest(self, new_pop):
        """Update personal best positions"""
        for i in range(len(new_pop)):
            # Compare new position with personal best
            if self._dominates(new_pop[i].F, self.pbest_f[i]):
                self.pbest[i] = new_pop[i].copy()
                self.pbest_f[i] = new_pop[i].F.copy()
            elif self._dominates(self.pbest_f[i], new_pop[i].F):
                # Keep current pbest
                pass
            else:
                # Non-dominated case: use crowding distance to decide
                # Combine both solutions to calculate crowding distance
                F_combined = np.vstack([self.pbest_f[i], new_pop[i].F])

                try:
                    crowding = self.cd.do(F_combined)
                    # Select the one with higher crowding distance (more diverse)
                    if crowding[1] > crowding[0]:  # new solution has higher crowding
                        self.pbest[i] = new_pop[i].copy()
                        self.pbest_f[i] = new_pop[i].F.copy()
                    # Otherwise keep current pbest
                except Exception:
                    # Fallback to random selection if crowding distance fails
                    if self.random_state.random() < 0.5:
                        self.pbest[i] = new_pop[i].copy()
                        self.pbest_f[i] = new_pop[i].F.copy()

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2"""
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _set_optimum(self, **kwargs):
        """Set the optimum solutions from the archive"""
        if len(self.archive) > 0:
            self.opt = self.archive.copy()
        else:
            self.opt = Population.empty()


# Parse docstring for documentation
parse_doc_string(MOPSO_CD.__init__)
