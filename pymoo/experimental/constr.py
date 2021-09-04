import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.cython.stochastic_ranking import stochastic_ranking
from pymoo.core.population import Population


def find_ranks(l, s=None, start=1):
    assert len(l.shape) == 1, "Please provide a vector of shape one!"

    if s is None:
        s = l.argsort()

    ranks = np.zeros_like(s)

    rank = start
    val = l[s[0]]

    for j in s:
        if val != l[j]:
            rank += 1
        ranks[j] = rank

    return ranks


def cv_rank_sum(G):
    _, n_constr = G.shape

    # the ranks to be returned later on
    ranks = []

    # for each constraint value provided
    for g in G.T:

        # find the rank for each constraint - least infeasible solution will have rank 1
        r = find_ranks(np.maximum(0, g))

        # if there is at least one feasible solution - let us start from rank 0
        if np.any(g <= 0):
            r = r - 1

        ranks.append(r)

    ranks = np.column_stack(ranks)

    # now sum the ranks of all the constraints up
    rank_sum = ranks.sum(axis=1).astype(float)

    return rank_sum


def rel_eps_constr(a, b, eps):
    f_a, f_b, cv_a, cv_b = a.F[0], b.F[0], a.CV[0], b.CV[0]

    if (cv_a <= eps and cv_b <= eps) or cv_a == cv_b:
        if f_a < f_b:
            return 1.0
        elif f_a == f_b:
            return 0.0
        else:
            return -1.0
    else:
        if cv_a < cv_b:
            return 1.0
        else:
            return -1.0


def eps_replacement(pop, off, eps):
    repl = np.full(len(pop), False)
    for k in range(len(pop)):

        if rel_eps_constr(pop[k], off[k], eps[k]) <= 0:
            repl[k] = True

    return repl


class EpsilonConstraintReplacement(ReplacementSurvival):

    def __init__(self, eps=0.001, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def _do(self, problem, pop, off, algorithm=None, **kwargs):
        n = len(pop)

        pop_X, pop_f, pop_cv = pop.get("X", "F", "CV")
        pop_f, pop_cv = pop_f[:, 0], pop_cv[:, 0]
        pop_feas = pop_cv <= 0

        infeas = np.sort(pop_cv[~pop_feas])
        n_infeas = len(pop) // 2

        if len(infeas) == 0:
            eps = np.inf
        else:
            k = min(n_infeas, len(infeas) - 1)
            eps = infeas[k]

        eps = eps * np.ones(n)

        return eps_replacement(pop, off, eps)


class FuzzyEpsilonConstraintReplacement(ReplacementSurvival):

    def __init__(self, eps=0.01, t=10, **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.eps = eps
        self.cnt = None

    def _do(self, problem, pop, off, algorithm=None, **kwargs):

        if self.cnt is None:
            self.cnt = np.zeros(len(pop), dtype=int)

        cnt = self.cnt

        cv = pop.get("CV")[:, 0]

        # cnt = self.cnt
        cnt = algorithm.n_gen - pop.get("n_gen") - 1

        # make sure we never replace the best solution if we would consider feasibility first
        best = FitnessSurvival().do(problem, Population.merge(pop, off), n_survive=1)[0]

        eps = np.zeros(len(pop))

        for k, t in enumerate(cnt):

            # cycle = (t // (4 * self.t))
            # max_eps = (2 ** cycle) * self.eps

            max_eps = self.eps

            t = t % (4 * self.t)

            if t < self.t:
                eps[k] = cv[k] + (max_eps - cv[k]) * (t / self.t)
            elif t < 2 * self.t:
                eps[k] = max_eps
            elif t < 3 * self.t:
                eps[k] = max_eps * (1 - ((t % self.t) / self.t))
            else:
                eps[k] = 0.0

        eps_is_zero = np.where(eps <= 0)[0]

        # print(len(eps_is_zero))

        repl = np.full(len(pop), False)
        for k in range(len(pop)):

            if pop[k] == best:
                repl[k] = False
            elif off[k] == best:
                repl[k] = True

            else:
                if rel_eps_constr(pop[k], off[k], eps[k]) <= 0:
                    repl[k] = True

        # self.cnt[repl] = 0
        # self.cnt[~repl] += 1

        return repl


class StochasticRankingImprovementReplacement(ReplacementSurvival):

    def _do(self, problem, pop, off, **kwargs):

        # merge the population for to find the ranks of them combined
        merged = Population.merge(pop, off)

        F, G = merged.get("F", "G")
        assert F.shape[1] == 1, "This survival only works for single-objective problems."
        f = F[:, 0]

        # for the constraint violation use the rank sum which makes the metric scale invariant
        cv = cv_rank_sum(G)
        # cv = merged.get("CV")[:, 0]

        merged.set("cv_rank_sum", cv)

        repl = np.full(len(pop), False)

        for k in range(len(pop)):
            a, b = pop[k], off[k]

            # cv_a, cv_b = a.get("cv_rank_sum"), b.get("cv_rank_sum")
            cv_a, cv_b = a.CV[0], b.CV[0]

            if np.random.random() < 0.25 or (cv_a == 0 and cv_b == 0):
                if b.F <= a.F:
                    repl[k] = True

            else:
                if cv_b <= cv_a:
                    repl[k] = True

        # make sure we never replace the best solution if we would consider feasibility first
        hierarch_sort = hierarchical_sort(f, cv)
        best = hierarch_sort[0]

        # if best from pop force not replacing it
        if best < len(pop):
            repl[best] = False

        # if best from offsprings force replacement
        else:
            repl[best - len(pop)] = True

        return repl

        # perform the stochastic sorting which gives infeasible solutions a chance to win over feasible once
        I = np.random.permutation(len(merged))
        S = I[stochastic_ranking(f, cv, 0.45, I=I)]

        # calculate the rank of each individual
        rank = S.argsort()

        rank_pop, rank_off = rank[:len(pop)], rank[len(pop):]
        repl = rank_off < rank_pop

        # make sure we never replace the best solution if we would consider feasibility first
        hierarch_sort = hierarchical_sort(f, cv)
        best = hierarch_sort[0]

        # if best from pop force not replacing it
        if best < len(pop):
            repl[best] = False

        # if best from offsprings force replacement
        else:
            repl[best - len(pop)] = True

        # set the rank to the all the individuals to be considered for later examples in an algorithm
        merged.set("sr", rank)

        return repl
