import numpy as np

from pymoo.util.dominator import Dominator


class NonDominatedRank:
    def __init__(self):
        pass

    @staticmethod
    def is_dominating(f, F):
        smaller = np.any(f < F, axis=1)
        larger = np.any(f > F, axis=1)
        dom = np.logical_and(smaller, np.logical_not(larger)) * 1 \
              + np.logical_and(larger, np.logical_not(smaller)) * -1
        return dom

    @staticmethod
    def get_non_dominated(F, F_, return_index=False):
        F_all = np.concatenate([F, F_], axis=0)
        front = NonDominatedRank().get_front(F_all)
        front = [i-F.shape[0] for i in front if i >= F.shape[0]]
        if return_index:
            return front
        return F_[front, :]

    @staticmethod
    def calc(F):
        fronts = NonDominatedRank.calc_as_fronts(F, None)
        return NonDominatedRank.calc_from_fronts(fronts)

    @staticmethod
    def calc_from_fronts(fronts):
        n = sum([len(f) for f in fronts])
        rank = np.zeros(n, dtype=np.int)
        for i in range(len(fronts)):
            for idx in fronts[i]:
                rank[idx] = i
        return rank

    @staticmethod
    def calc_as_fronts_naive(F, G):

        M = Dominator.calc_domination_matrix(F, G)
        fronts = []

        remaining = set(range(F.shape[0]))

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

    @staticmethod
    def get_front(F, G=None):
        return NonDominatedRank.calc_as_fronts(F, G, only_pareto_front=True)

    @staticmethod
    def calc_as_fronts(F, G=None, only_pareto_front=False):

        # calculate the dominance matrix
        n = F.shape[0]
        M = Dominator.calc_domination_matrix(F, G)

        fronts = []

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

        if only_pareto_front:
            return current_front

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
