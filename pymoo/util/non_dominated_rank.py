import numpy as np
import pygmo as pg
from scipy.spatial.distance import squareform, pdist

from pymoo.util.dominator import Dominator


class NonDominatedRank:
    def __init__(self):
        pass

    @staticmethod
    def calc(pop):
        fronts = NonDominatedRank.calc_as_fronts(pop)
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
    def calc_as_fronts_naive(pop):

        fronts = []

        left_over = list(range(len(pop)))

        while len(left_over) > 0:

            front = []

            for i in left_over:

                is_dominated = False
                dominating = set()

                for j in front:
                    rel = Dominator.get_relation(pop[i], pop[j])
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

            left_over = [e for e in left_over if e not in front]
            fronts.append(front)

        return fronts



    @staticmethod
    def calc_as_fronts(F, G):

        # calculate the dominance matrix
        n = F.shape[0]
        M = Dominator.calc_domination_matrix(F,G)

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
