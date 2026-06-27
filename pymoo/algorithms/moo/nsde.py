"""NSDE - NSGA-II with differential evolution operators."""

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.infill import InfillCriterion
from pymoo.core.population import Population
from pymoo.operators.crossover.binx import mut_binomial
from pymoo.operators.crossover.dex import DE_REPAIRS
from pymoo.operators.crossover.expx import mut_exp
from pymoo.operators.mutation.nom import NoMutation
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.selection.des import DES
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding


# =========================================================================================================
# VariantDE — DE mating operator used by NSDE-family algorithms
# =========================================================================================================


class VariantDE(InfillCriterion):
    def __init__(
        self,
        variant="DE/rand/1/bin",
        CR=0.7,
        F=None,
        gamma=1e-4,
        de_repair="bounce-back",
        mutation=None,
        **kwargs,
    ):

        super().__init__(eliminate_duplicates=None, **kwargs)

        _, selection_variant, n_diff, crossover_variant = variant.split("/")
        n_diffs = int(n_diff)
        if "-to-" in variant:
            n_diffs += 1

        self.selection = DES(selection_variant)
        # total parents: 1 target + 1 mutation base + 2*(n_diffs-1) additional diff vectors
        # = 2 + 2*n_diffs (matches PR's DEX.n_parents = 2 + 2*n_diffs)
        self.n_total_parents = 2 + 2 * n_diffs
        self.n_diffs = n_diffs
        self.crossover_variant = crossover_variant
        self.CR = CR
        self.gamma = gamma

        if F is None:
            F = (0.0, 1.0)
        self.F = F

        if callable(de_repair):
            self.de_repair = de_repair
        else:
            if de_repair not in DE_REPAIRS:
                raise KeyError(
                    f"de_repair must be callable or one of {list(DE_REPAIRS.keys())}"
                )
            self.de_repair = DE_REPAIRS[de_repair]

        self.mutation = mutation if mutation is not None else NoMutation()

    def _do(self, problem, pop, n_offsprings, random_state=None, **kwargs):
        n_var = problem.n_var

        # Select parents: shape [n_offsprings, n_total_parents]
        P = self.selection.do(
            problem,
            pop,
            n_offsprings,
            self.n_total_parents,
            to_pop=False,
            random_state=random_state,
        )

        X = pop.get("X")
        Xr = X[P]  # [n_offsprings, n_total_parents, n_var]

        # Mutation base is column 1 (column 0 is the target for crossover)
        V = Xr[:, 1].copy()

        # Difference pairs start at column 2: (col2,col3), (col4,col5), ...
        for k in range(self.n_diffs):
            ai, bi = 2 + 2 * k, 3 + 2 * k
            F = self._sample_F(n_offsprings, random_state)
            if self.gamma is not None:
                F_mat = F[:, None] * (
                    1 + self.gamma * (random_state.random((n_offsprings, n_var)) - 0.5)
                )
                V = V + F_mat * (Xr[:, ai] - Xr[:, bi])
            else:
                V = V + F[:, None] * (Xr[:, ai] - Xr[:, bi])

        # Repair donor vector if bounds are violated
        if problem.has_bounds():
            V = self.de_repair(
                V, Xr[:, 1], *problem.bounds(), random_state=random_state
            )

        # Crossover: trial vector from target (col 0) and donor V
        Xi = Xr[:, 0]
        if self.crossover_variant == "bin":
            M = mut_binomial(
                n_offsprings,
                n_var,
                self.CR,
                at_least_once=True,
                random_state=random_state,
            )
        elif self.crossover_variant == "exp":
            M = mut_exp(
                n_offsprings,
                n_var,
                self.CR,
                at_least_once=True,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown crossover variant: {self.crossover_variant}")

        trial = Xi.copy()
        trial[M] = V[M]

        off = Population.new("X", trial)

        # Optional posterior mutation
        off = self.mutation(problem, off, random_state=random_state)

        return off

    def _sample_F(self, n, random_state):
        if hasattr(self.F, "__iter__"):
            lo, hi = self.F[0], self.F[1]
            return lo + random_state.random(n) * (hi - lo)
        return np.full(n, self.F)


# =========================================================================================================
# NSDE
# =========================================================================================================


class NSDE(NSGA2):
    def __init__(
        self,
        pop_size=100,
        sampling=None,
        variant="DE/rand/1/bin",
        CR=0.7,
        F=None,
        gamma=1e-4,
        de_repair="bounce-back",
        survival=None,
        **kwargs,
    ):
        """NSDE combines NSGA-II sorting and survival with DE mutation and crossover.

        For many-objective problems, try NSDE-R, GDE3-MNN, or GDE3-2NN.
        For bi-objective problems, survival=RankAndCrowding(crowding_func='pcd') is effective.

        Args:
            pop_size: Population size. Defaults to 100.
            sampling: Sampling strategy. Defaults to LHS().
            variant: DE strategy string: "DE/selection/n/crossover".
                selection: 'rand', 'best', 'current-to-best', 'current-to-rand', 'ranked'.
                crossover: 'bin' or 'exp'. Defaults to 'DE/rand/1/bin'.
            CR: Crossover rate in [0, 1]. Defaults to 0.7.
            F: Scale factor(s) in (0, 2]. Defaults to randomized in (0, 1).
            gamma: Jitter deviation. Defaults to 1e-4.
            de_repair: Repair for DE donor vectors: 'bounce-back', 'midway', 'rand-init', 'to-bounds'.
            survival: Survival strategy. Defaults to RankAndCrowding().
            **kwargs: Additional keyword arguments passed to parent NSGA2.
        """
        if sampling is None:
            sampling = LHS()
        if survival is None:
            survival = RankAndCrowding()

        mating = VariantDE(
            variant=variant, CR=CR, F=F, gamma=gamma, de_repair=de_repair
        )

        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            mating=mating,
            survival=survival,
            eliminate_duplicates=None,
            n_offsprings=pop_size,
            **kwargs,
        )
