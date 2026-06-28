"""Omni-Optimizer: a generic NSGA-II based algorithm for single/multi-objective and single/multi-modal optimization (Deb & Tiwari, 2008)."""

import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.core.selection import Selection
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.misc import has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort

# A finite value (larger than any normalized crowding distance) assigned to boundary
# solutions. The original implementation deliberately uses a finite sentinel instead of
# infinity so that the average crowding distance used to combine the objective- and
# variable-space metrics remains well defined.
BOUNDARY = 10.0


# =========================================================================================================
# Epsilon (loose) dominance with a dynamically calculated epsilon
# =========================================================================================================


class LooseDominator:
    """Modified (loose) epsilon-dominance used by the Omni-Optimizer [1]_.

    A solution ``a`` is said to loosely dominate ``b`` only if it dominates ``b`` in the
    usual Pareto sense *and* is better by more than a margin ``delta * epsilon_j`` in at
    least one objective ``j``. The per-objective epsilon is calculated dynamically from
    the population that is being sorted as the range of each objective::

        epsilon_j = max_j(F) - min_j(F)

    Solutions that are closer than ``delta * epsilon_j`` in every objective are therefore
    treated as mutually non-dominated and end up in the same front. Together with the
    variable-space crowding distance this is what allows the Omni-Optimizer to maintain
    multiple equivalent (in objective space) solutions.

    This class follows the ``calc_domination_matrix`` interface so it can be plugged into
    :class:`~pymoo.util.nds.non_dominated_sorting.NonDominatedSorting` via its
    ``dominator`` argument.

    Parameters
    ----------
    delta : float
        Fraction of the per-objective range used as the epsilon margin. Defaults to 0.001.

    References:
    ----------
    .. [1] K. Deb and S. Tiwari, "Omni-optimizer: A generic evolutionary algorithm for
       single and multi-objective optimization", European Journal of Operational Research,
       185(3), 2008, pp. 1062-1087.
    """

    def __init__(self, delta=0.001):
        self.delta = delta

    def calc_domination_matrix(self, F, _F=None):
        if _F is None:
            _F = F

        n, m = F.shape[0], _F.shape[0]

        # epsilon is calculated dynamically as a fraction of the range of each objective
        epsilon = self.delta * (F.max(axis=0) - F.min(axis=0))

        # build all pairwise combinations (i-th block compares F[i] against every _F)
        L = np.repeat(F, m, axis=0)
        R = np.tile(_F, (n, 1))

        # usual Pareto relation: is the left solution better / worse in any objective?
        better = np.any(L < R, axis=1).reshape(n, m)
        worse = np.any(L > R, axis=1).reshape(n, m)

        # the left solution dominates / is dominated in the usual sense
        dominates = better & ~worse
        dominated = worse & ~better

        # the relation only counts if the margin is exceeded in at least one objective
        better_by_eps = np.any(L + epsilon < R, axis=1).reshape(n, m)
        worse_by_eps = np.any(L > R + epsilon, axis=1).reshape(n, m)

        M = (dominates & better_by_eps).astype(int) - (dominated & worse_by_eps).astype(int)
        return M


# =========================================================================================================
# Crowding distance in objective and variable space
# =========================================================================================================


def calc_crowding_distance_in_space(Y, space="objective"):
    """Crowding distance of a single front computed in one space.

    This is the NSGA-II crowding distance (sum of the normalized distances to the nearest
    neighbors along each dimension), with two characteristics of the Omni-Optimizer [1]_:

    - the contribution is averaged over the number of dimensions, and
    - boundary solutions are handled differently in objective and variable space.

    In objective space the extreme solutions of each objective receive the (finite)
    :data:`BOUNDARY` value so that the best solution of every objective is preserved. In
    variable space no solution is treated as infinitely important; instead the boundary
    solutions receive twice the distance to their only neighbor, mirroring the reference
    implementation.

    Parameters
    ----------
    Y : numpy.ndarray
        ``(n, d)`` matrix of either objective values or decision variables of the front.
    space : str
        Either ``"objective"`` or ``"variable"``.

    References:
    ----------
    .. [1] K. Deb and S. Tiwari, "Omni-optimizer: A generic evolutionary algorithm for
       single and multi-objective optimization", European Journal of Operational Research,
       185(3), 2008, pp. 1062-1087.
    """
    n, d = Y.shape

    # for one or two solutions every solution is a boundary solution
    if n <= 2:
        return np.full(n, BOUNDARY)

    cd = np.zeros(n)
    is_boundary = np.zeros(n, dtype=bool)

    for j in range(d):
        order = np.argsort(Y[:, j], kind="mergesort")
        lo, hi = order[0], order[-1]
        span = Y[hi, j] - Y[lo, j]

        if space == "objective":
            # the best (minimum) solution of this objective is a boundary solution
            is_boundary[lo] = True
            if span != 0:
                interior = order[1:-1]
                cd[interior] += (Y[order[2:], j] - Y[order[:-2], j]) / span
        else:
            if span != 0:
                # the boundary solutions get twice the gap to their single neighbor
                cd[lo] += 2.0 * (Y[order[1], j] - Y[lo, j]) / span
                cd[hi] += 2.0 * (Y[hi, j] - Y[order[-2], j]) / span
                interior = order[1:-1]
                cd[interior] += (Y[order[2:], j] - Y[order[:-2], j]) / span

    cd /= d

    if space == "objective":
        cd[is_boundary] = BOUNDARY

    return cd


def calc_omni_crowding_distance(F, X, obj_crowding=True, var_crowding=True):
    """Combined objective- and variable-space crowding distance of a single front.

    The crowding distance is computed independently in objective and variable space
    (see :func:`calc_crowding_distance_in_space`). For every solution, if it is less
    crowded than the average of the front in *either* space the larger of the two values
    is assigned, otherwise the smaller one is used. This rewards solutions that maintain
    diversity in at least one of the two spaces.

    Parameters
    ----------
    F : numpy.ndarray
        Objective values of the front, ``(n, n_obj)``.
    X : numpy.ndarray
        Decision variables of the front, ``(n, n_var)``.
    obj_crowding, var_crowding : bool
        Whether to use the objective- and/or variable-space niching. At least one of them
        must be enabled. Disabling the variable-space niching recovers the NSGA-II
        behavior; disabling the objective-space niching niches purely in variable space.
    """
    if not (obj_crowding or var_crowding):
        raise ValueError("At least one of objective- or variable-space crowding must be enabled.")

    n = len(F)

    obj_cd = calc_crowding_distance_in_space(F, space="objective") if obj_crowding else None
    var_cd = calc_crowding_distance_in_space(X, space="variable") if var_crowding else None

    # only a single space is used
    if not var_crowding:
        return obj_cd
    if not obj_crowding:
        return var_cd

    n_obj, n_var = F.shape[1], X.shape[1]

    # the average crowding distance of the front excluding boundary solutions in obj. space
    avg_obj = obj_cd[obj_cd != BOUNDARY].sum() / n / n_obj
    avg_var = var_cd.sum() / n / n_var

    take_max = (obj_cd > avg_obj) | (var_cd > avg_var)

    cd = np.where(take_max, np.maximum(obj_cd, var_cd), np.minimum(obj_cd, var_cd))
    return cd


# =========================================================================================================
# Survival
# =========================================================================================================


class OmniOptimizerSurvival(Survival):
    """Rank and (objective + variable space) crowding survival of the Omni-Optimizer [1]_.

    The non-dominated sorting uses the dynamically calculated epsilon (loose) dominance
    (:class:`LooseDominator`) and the last surviving front is truncated by the combined
    objective- and variable-space crowding distance (:func:`calc_omni_crowding_distance`).

    Parameters
    ----------
    delta : float
        Epsilon margin (fraction of each objective's range) for the loose dominance.
    obj_crowding, var_crowding : bool
        Whether to niche in objective and/or variable space.

    References:
    ----------
    .. [1] K. Deb and S. Tiwari, "Omni-optimizer: A generic evolutionary algorithm for
       single and multi-objective optimization", European Journal of Operational Research,
       185(3), 2008, pp. 1062-1087.
    """

    def __init__(self, delta=0.001, obj_crowding=True, var_crowding=True):
        super().__init__(filter_infeasible=True)
        self.delta = delta
        self.obj_crowding = obj_crowding
        self.var_crowding = var_crowding
        self.nds = NonDominatedSorting(dominator=LooseDominator(delta=delta))

    def _do(self, problem, pop, *args, n_survive=None, random_state=None, **kwargs):

        # objective values and decision variables of the (feasible) population
        F = pop.get("F").astype(float, copy=False)
        X = pop.get("X").astype(float, copy=False)

        survivors = []

        # non-dominated sorting using the dynamic epsilon (loose) dominance
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):
            # combined objective- and variable-space crowding distance of the front
            crowding_of_front = calc_omni_crowding_distance(
                F[front, :],
                X[front, :],
                obj_crowding=self.obj_crowding,
                var_crowding=self.var_crowding,
            )

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                idx = randomized_argsort(
                    crowding_of_front,
                    order="descending",
                    method="numpy",
                    random_state=random_state,
                )
                idx = idx[: (n_survive - len(survivors))]

            # otherwise take the whole front
            else:
                idx = np.arange(len(front))

            survivors.extend(front[idx])

        return pop[survivors]


# =========================================================================================================
# Restricted (nearest neighbor) mating selection
# =========================================================================================================


class NeighborBasedTournamentSelection(Selection):
    """Restricted binary tournament selection of the Omni-Optimizer [1]_.

    Instead of pairing two random solutions, each tournament is held between a randomly
    drawn solution and its nearest neighbor in the (normalized) decision space. The two
    competitors are removed from the pool, so that every solution participates in exactly
    one tournament per pass over the population. The comparison itself is the usual
    NSGA-II crowded-comparison (Pareto dominance, then crowding distance, then random).

    Restricting the mating to nearby solutions biases recombination towards the same
    region of the decision space, which helps to preserve distinct (but equivalent)
    optima.

    Parameters
    ----------
    func_comp : callable
        The binary tournament comparison. Defaults to NSGA-II's ``binary_tournament``.

    References:
    ----------
    .. [1] K. Deb and S. Tiwari, "Omni-optimizer: A generic evolutionary algorithm for
       single and multi-objective optimization", European Journal of Operational Research,
       185(3), 2008, pp. 1062-1087.
    """

    def __init__(self, func_comp=binary_tournament, **kwargs):
        super().__init__(**kwargs)
        self.func_comp = func_comp

    def _do(self, problem, pop, n_select, n_parents, random_state=None, **kwargs):

        n_winners = n_select * n_parents
        n = len(pop)

        # normalize the decision space so that every variable contributes equally to the
        # distance used to determine the nearest neighbor
        X = pop.get("X").astype(float, copy=False)
        xl, xu = X.min(axis=0), X.max(axis=0)
        norm = xu - xl
        norm[norm == 0] = 1.0
        Xn = (X - xl) / norm

        # collect (solution, nearest neighbor) pairs to compete against each other
        pairs = np.empty((n_winners, 2), dtype=int)
        count = 0

        while count < n_winners:
            # a fresh random order of all solutions for this pass
            remaining = list(random_state.permutation(n))

            while len(remaining) >= 2 and count < n_winners:
                # the first (randomly drawn) solution of the pass
                p = remaining.pop(0)

                # its nearest neighbor in normalized decision space among the remaining
                rest = np.array(remaining)
                dist = np.sum((Xn[rest] - Xn[p]) ** 2, axis=1)
                nn = int(np.argmin(dist))
                q = remaining.pop(nn)

                pairs[count] = (p, q)
                count += 1

        # run the binary tournaments
        S = self.func_comp(pop, pairs, random_state=random_state, **kwargs)

        return np.reshape(S, (n_select, n_parents))


# =========================================================================================================
# Algorithm
# =========================================================================================================


class OmniOptimizer(GeneticAlgorithm):
    def __init__(
        self,
        pop_size=100,
        delta=0.001,
        obj_crowding=True,
        var_crowding=True,
        sampling=LHS(),
        selection=NeighborBasedTournamentSelection(func_comp=binary_tournament),
        crossover=SBX(eta=20, prob=0.8),
        mutation=PM(eta=20),
        survival=None,
        output=MultiObjectiveOutput(),
        **kwargs,
    ):
        """Omni-Optimizer: a generic evolutionary algorithm for single/multi-objective optimization.

        Proposed by Deb and Tiwari (*European Journal of Operational Research*, 185(3),
        2008, pp. 1062-1087) for single- and multi-objective, single- and multi-global
        optimization. It is an NSGA-II based algorithm with three distinctive components:

        - a non-dominated sorting based on a *loose* epsilon-dominance whose epsilon is
          calculated dynamically from the population (:class:`LooseDominator`),
        - a crowding distance computed in *both* the objective and the variable space
          (:func:`calc_omni_crowding_distance`), and
        - a restricted binary tournament selection between a solution and its nearest
          neighbor in the decision space (:class:`NeighborBasedTournamentSelection`).

        These components allow the algorithm to find and maintain multiple equivalent
        Pareto-optimal solutions, i.e. solutions that map to (almost) the same point in
        objective space but are distinct in decision space.

        Args:
            pop_size: The population size.
            delta: The epsilon margin (as a fraction of each objective's range) used for
                the loose dominance. ``delta=0`` recovers the usual Pareto dominance.
            obj_crowding: Whether to niche in the objective space.
            var_crowding: Whether to niche in the variable space. Disabling it essentially
                recovers NSGA-II.
            sampling: Sampling operator (defaults to the operator from the original paper).
            selection: Selection operator (defaults to the operator from the original paper).
            crossover: Crossover operator (defaults to SBX, as in the original paper).
            mutation: Mutation operator (defaults to polynomial mutation, as in the paper).
            survival: Survival operator (defaults to :class:`OmniOptimizerSurvival`).
            output: Display output used during the optimization run.
            **kwargs: Additional keyword arguments passed to the genetic algorithm base class.
        """
        if survival is None:
            survival = OmniOptimizerSurvival(delta=delta, obj_crowding=obj_crowding, var_crowding=var_crowding)

        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            **kwargs,
        )

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = "comp_by_dom_and_crowding"

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]


parse_doc_string(OmniOptimizer.__init__)
