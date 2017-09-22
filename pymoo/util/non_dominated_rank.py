import numpy as np
import pygmo as pg
from util.dominator import Dominator


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
        rank = np.zeros(n)
        for i in range(len(fronts)):
            for idx in fronts[i]:
                rank[idx] = i
        return rank

    @staticmethod
    def calc_as_fronts_pygmo(pop):

        n = len(pop)
        if n == 0:
            return np.array()
        m = len(pop[0].f)

        objectives = [pop[i].f for i in range(n)]
        constr = np.array([Dominator.get_constraint_violation(pop[i]) for i in range(n)])
        f_max = np.array([max([objectives[i][j] for i in range(n)]) for j in range(m)])

        for i in range(len(constr)):
            if constr[i] > 0:
                objectives[i] = f_max + constr[i]

        return pg.fast_non_dominated_sorting(objectives)[0]

    @staticmethod
    def calc_as_fronts(pop):
        fronts = []

        # number of individuals
        n = len(pop)

        # final rank that will be returned
        ranked = np.zeros(n)

        # for each individual a list of all individuals that are dominated by this one
        is_dominating = [[] for _ in range(n)]

        # storage for the number of solutions dominated this one
        n_dominated = np.zeros(n)

        current_front = []

        for i in range(n):

            for j in range(i+1, n):
                rel = Dominator.get_relation(pop[i], pop[j])
                if rel == 1:
                    is_dominating[i].append(j)
                    n_dominated[j] += 1
                elif rel == -1:
                    is_dominating[j].append(i)
                    n_dominated[i] += 1

            if n_dominated[i] == 0:
                current_front.append(i)
                ranked[i] = 1.0

        # append the first front to the current front
        fronts.append(current_front)

        # while not all solutions are assigned to a pareto front
        while np.sum(ranked) != n:

            next_front = []

            # for each individual in the current front
            for i in current_front:

                # all solutions that are dominated by this individuals
                for j in is_dominating[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
                        ranked[j] = 1.0

            fronts.append(next_front)
            current_front = next_front

        return fronts
