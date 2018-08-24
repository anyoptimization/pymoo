import numpy as np

from pymoo.util.dominator import Dominator


class NonDominatedRank:

    def __init__(self, epsilon=0.0) -> None:
        super().__init__()
        self.epsilon = epsilon

    def do(self, F, return_rank=False, only_non_dominated_front=False, n_stop_if_exceed=None):

        # calculate the domination matrix - faster because vectorized
        M = Dominator.calc_domination_matrix(F, epsilon=self.epsilon)

        # get the fronts using that matrix
        fronts = fast_non_dominated_sort(M)

        # convert to numpy array for each front and filter by n_stop_if_ranked if desired
        _fronts = []
        ranked = 0
        for front in fronts:

            # stop if more than this solutions are ranked
            if n_stop_if_exceed is not None and ranked > n_stop_if_exceed:
                break
            else:
                _fronts.append(np.array(front, dtype=np.int))

            # increment the ranked solution counter
            ranked += len(front)

        fronts = _fronts

        if only_non_dominated_front:
            return fronts[0]

        if return_rank:
            rank = np.full(M.shape[0], np.inf, dtype=np.int)
            for i, front in enumerate(fronts):
                rank[front] = i
            return fronts, rank

        return fronts


# Returns all indices of F that are not dominated by the other objective values
def find_non_dominated(F, _F=None):
    M = Dominator.calc_domination_matrix(F, _F)
    I = np.where(np.all(M >= 0, axis=1))[0]
    return I


def non_dominated_sort_naive(M):
    fronts = []
    remaining = set(range(M.shape[0]))

    while len(remaining) > 0:

        front = []

        for i in remaining:

            is_dominated = False
            dominating = set()

            for j in front:
                rel = M[i, j]
                if rel == 1:
                    dominating.add(j)
                elif rel == -1:
                    is_dominated = True
                    break

            if is_dominated:
                continue
            else:
                front = [x for x in front if x not in dominating]
                front.append(i)

        [remaining.remove(e) for e in front]
        fronts.append(front)

    return fronts


def fast_non_dominated_sort(M):
    # calculate the dominance matrix
    n = M.shape[0]

    fronts = []

    if n == 0:
        return fronts

    # final rank that will be returned
    n_ranked = 0
    ranked = np.zeros(n, dtype=np.int)

    # for each individual a list of all individuals that are dominated by this one
    is_dominating = [[] for _ in range(n)]

    # storage for the number of solutions dominated this one
    n_dominated = np.zeros(n)

    current_front = []

    for i in range(n):

        for j in range(i + 1, n):
            rel = M[i, j]
            if rel == 1:
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif rel == -1:
                is_dominating[j].append(i)
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            current_front.append(i)
            ranked[i] = 1.0
            n_ranked += 1

    # append the first front to the current front
    fronts.append(current_front)

    # while not all solutions are assigned to a pareto front
    while n_ranked < n:

        next_front = []

        # for each individual in the current front
        for i in current_front:

            # all solutions that are dominated by this individuals
            for j in is_dominating[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1.0
                    n_ranked += 1

        fronts.append(next_front)
        current_front = next_front

    return fronts
