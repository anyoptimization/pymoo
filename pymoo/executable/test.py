import numpy as np

from non_dominated_sorting import get_relation, best_order_sort
from pymoo.util.non_dominated_sorting import NonDominatedSorting, rank_from_fronts


def best_order_sort(F):
    n_points, n_obj = F.shape

    Q = np.argsort(F, axis=0)

    rank = np.full(n_points, -1, dtype=np.int)

    fronts = []
    L = [[] for _ in range(n_obj)]
    n_fronts = 0
    n_ranked = 0

    # the outer loop iterates through all solutions
    for i in range(n_points):

        # the inner loop through each objective values (sorted)
        for j in range(n_obj):

            # index of the current solution
            s = Q[i, j]

            # if solution was already ranked before - just append it to the corresponding front
            if rank[s] != -1:
                L[j][rank[s]].append(s)

            # otherwise we rank it for the first time
            else:

                # the rank of this solution is stored here
                s_rank = -1

                # for each front ranked for this objective
                for k in range(n_fronts):

                    is_dominated = False

                    # for each entry in that front
                    for e in L[j][k]:
                        is_dominated = get_relation(F, s, e) == -1

                        # if just one solution dominates the current one - go to the next front
                        if is_dominated:
                            break

                    # if no solutions in the front dominates this one we found the rank
                    if not is_dominated:
                        s_rank = k
                        break

                # we need to add a new front for each objective
                if s_rank == -1:
                    s_rank = n_fronts
                    n_fronts += 1

                    fronts.append([])
                    for l in range(n_obj):
                        L[l].append([])

                L[j][s_rank].append(s)
                fronts[s_rank].append(s)
                rank[s] = s_rank
                n_ranked += 1

        if n_ranked == n_points:
            break

    return fronts


if __name__ == '__main__':
    np.random.seed(1)

    F = np.random.random((200, 2))

    fronts, rank = NonDominatedSorting().do(F, return_rank=True)
    _fronts = best_order_sort(F)

    print(np.all(rank == rank_from_fronts(_fronts)))

