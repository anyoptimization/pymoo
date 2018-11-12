import numpy as np

from pymoo.util.non_dominated_sorting import NonDominatedSorting, rank_from_fronts


def _is_dominating_or_equal(F, a, b, C, k):
    for i in range(k, C.shape[1]):
        if F[b, C[b, i]] < F[a, C[b, i]]:
            return False
    return True


def _best_order_sort(F):
    n_points, n_obj = F.shape

    Q = np.zeros(F.shape, dtype=np.int)
    Q[:, 0] = np.lexsort(F[:, ::-1].T, axis=0)
    for j in range(1, n_obj):
        Q[:, j] = np.lexsort(np.vstack([Q[:, 0], F[:, j]]), axis=0)

    counter = np.zeros(n_points, dtype=np.int)
    C = np.full(F.shape, -1)
    for i in range(n_points):
        for j in range(n_obj):
            s = Q[i, j]
            C[s, counter[s]] = j
            counter[s] += 1
    counter = np.zeros(n_points, dtype=np.int)

    check_if_equal = np.full(n_points, -1, dtype=np.int)

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
            s_next = Q[i + 1, j]

            # increase the counter for comparing this objective
            counter[s] += 1

            # if not the last solution
            if i < n_points - 1:
                if check_if_equal[s] == -1:
                    check_if_equal[s] = s_next
                elif check_if_equal[s] != s_next:
                    check_if_equal[s] = -2

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

                        is_dominated = _is_dominating_or_equal(F, e, s, C, counter[s])

                        if is_dominated and check_if_equal[e] == s:
                            is_equal = True
                            for o in range(n_obj):
                                if F[s, o] != F[e, o]:
                                    is_equal = False
                                    break
                            if is_equal:
                                is_dominated = False

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

    F = np.random.random((200, 3))
    F[20, :] = F[82, :]

    fronts, rank = NonDominatedSorting(method="cython_fast_best_order_sort").do(F, return_rank=True)
    _fronts = _best_order_sort(F)

    print(np.all(rank == rank_from_fronts(_fronts, F.shape[0])))
