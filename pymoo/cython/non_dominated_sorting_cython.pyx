# distutils: language = c++
# cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True


cimport cython
import numpy as np
cimport numpy as cnp
from libcpp cimport bool
from libcpp.vector cimport vector


cdef extern from "limits.h":
    int INT_MAX

def fast_non_dominated_sort(double[:,:] F, double epsilon = 0.0, int n_stop_if_ranked=INT_MAX):
    return c_fast_non_dominated_sort(F, epsilon, n_stop_if_ranked)

def best_order_sort(double[:,:] F):
    return c_best_order_sort(F)

def get_relation(F, a, b):
    return c_get_relation(F, a, b)

def fast_best_order_sort(double[:,:] F):
    return c_fast_best_order_sort(F)



cdef vector[vector[int]] c_fast_non_dominated_sort(double[:,:] F, double epsilon = 0.0, int n_stop_if_ranked=INT_MAX):

    cdef:
        int n_points, i, j, rel, n_ranked
        vector[int] current_front, next_front, n_dominated
        vector[vector[int]] fronts, is_dominating

    # calculate the dominance matrix
    n_points = F.shape[0]

    fronts = vector[vector[int]]()

    if n_points == 0:
        return fronts

    # final rank that will be returned
    n_ranked = 0

    # for each individual a list of all individuals that are dominated by this one
    is_dominating = vector[vector[int]](n_points)
    for _ in range(n_points):
        is_dominating.push_back(vector[int](n_points))

    n_dominated = vector[int]()
    for i in range(n_points):
        n_dominated.push_back(0)

    current_front = vector[int]()

    for i in range(n_points):

        for j in range(i + 1, n_points):

            rel = c_get_relation(F, i, j, epsilon)

            if rel == 1:
                is_dominating[i].push_back(j)
                n_dominated[j] += 1

            elif rel == -1:
                is_dominating[j].push_back(i)
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            current_front.push_back(i)
            n_ranked += 1

    # append the first front to the current front
    fronts.push_back(current_front)

    # while not all solutions are assigned to a pareto front or we can stop early because of stop criterium
    while (n_ranked < n_points) and (n_ranked < n_stop_if_ranked):

        next_front = vector[int]()

        # for each individual in the current front
        for i in current_front:

            # all solutions that are dominated by this individuals
            for j in is_dominating[i]:

                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.push_back(j)
                    n_ranked += 1

        fronts.push_back(next_front)
        current_front = next_front

    return fronts


cdef vector[vector[int]] c_best_order_sort(double[:,:] F):

    cdef:
        int n_points, n_obj, n_fronts, n_ranked, i, j, s, e, l, z
        vector[int] rank
        int[:,:] Q
        bool is_dominated
        vector[vector[int]] fronts, empty
        vector[vector[vector[int]]] L

    n_points = F.shape[0]
    n_obj = F.shape[1]
    fronts = vector[vector[int]]()

    _Q = np.zeros((n_points, n_obj), dtype=np.intc)
    for j in range(n_obj):
        _Q[:, j] = np.lexsort(F[:, j:][:, ::-1].T, axis=0)
    Q = _Q[:,:]

    rank = vector[int]()
    for i in range(n_points):
        rank.push_back(-1)

    L = vector[vector[vector[int]]]()
    for j in range(n_obj):
        empty = vector[vector[int]]()
        L.push_back(empty)

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
                L[j][rank[s]].push_back(s)

            # otherwise we rank it for the first time
            else:

                # the rank of this solution is stored here
                s_rank = -1

                # for each front ranked for this objective
                for k in range(n_fronts):

                    is_dominated = False

                    # for each entry in that front
                    for e in L[j][k]:

                        is_dominated = c_get_relation(F, s, e) == -1

                        # if one solution dominates the current one - go to the next front
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

                    fronts.push_back(vector[int]())
                    for l in range(n_obj):
                        L[l].push_back(vector[int]())

                L[j][s_rank].push_back(s)
                fronts[s_rank].push_back(s)
                rank[s] = s_rank
                n_ranked += 1

                if n_ranked == n_points:
                    break

    return fronts


cdef vector[vector[int]] c_fast_best_order_sort(double[:,:] F):

    cdef:
        int n_points, n_obj, n_fronts, n_ranked, i, j, s, e, l, z, s_next
        vector[int] rank, counter, check_if_equal
        int[:,:] Q
        bool is_dominated
        vector[vector[int]] fronts, empty ,C
        vector[vector[vector[int]]] L

    n_points = F.shape[0]
    n_obj = F.shape[1]
    fronts = vector[vector[int]]()


    _Q = np.zeros((n_points, n_obj), dtype=np.intc)
    _Q[:, 0] = np.lexsort(F[:, ::-1].T, axis=0)
    for j in range(1, n_obj):
        _Q[:, j] = np.lexsort(np.vstack([_Q[:, 0], F[:, j]]), axis=0)
    Q = _Q[:,:]


    counter = vector[int](n_points)
    C = vector[vector[int]]()
    for j in range(n_points):
        C.push_back(vector[int](n_obj))
    for i in range(n_points):
        for j in range(n_obj):
            s = Q[i, j]
            C[s][counter[s]] = j
            counter[s] += 1
    counter = vector[int](n_points)

    check_if_equal = vector[int]()
    for j in range(n_points):
        check_if_equal.push_back(-1)

    rank = vector[int]()
    for i in range(n_points):
        rank.push_back(-1)

    L = vector[vector[vector[int]]]()
    for j in range(n_obj):
        empty = vector[vector[int]]()
        L.push_back(empty)

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
                L[j][rank[s]].push_back(s)

            # otherwise we rank it for the first time
            else:

                # the rank of this solution is stored here
                s_rank = -1

                # for each front ranked for this objective
                for k in range(n_fronts):

                    # just necessary if no fronts exists
                    is_dominated = False
                    is_equal = False

                    # for each entry in that front
                    for e in L[j][k]:

                        # get the domination relation - might return true even if equal
                        is_dominated = c_is_dominating_or_equal(F, e, s, C, counter[s])

                        if is_dominated and check_if_equal[e] == s:
                            is_equal = c_is_equal(F, s, e)

                        # if just one solution dominates the current one - go to the next front
                        if is_dominated or is_equal:
                            break

                    # if no solutions in the front dominates this one we found the rank
                    if not is_dominated or is_equal:
                        s_rank = k
                        break

                # we need to add a new front for each objective
                if s_rank == -1:

                    s_rank = n_fronts
                    n_fronts += 1

                    fronts.push_back(vector[int]())
                    for l in range(n_obj):
                        L[l].push_back(vector[int]())

                L[j][s_rank].push_back(s)
                fronts[s_rank].push_back(s)
                rank[s] = s_rank
                n_ranked += 1

                if n_ranked == n_points:
                    break

    return fronts


cdef int c_get_relation(double[:,:] F, int a, int b, double epsilon = 0.0):

    cdef int size = F.shape[1]
    cdef int val
    cdef int i

    val = 0

    for i in range(size):

        if F[a,i] + epsilon < F[b,i]:
            # indifferent because once better and once worse
            if val == -1:
                return 0
            val = 1

        elif F[a,i] > F[b,i] + epsilon:

            # indifferent because once better and once worse
            if val == 1:
                return 0
            val = -1

    return val



cdef bool c_is_dominating_or_equal(double[:,:] F, int a, int b, vector[vector[int]]& C, int k):
    cdef int i, j
    for i in range(k, C[0].size()):
        j = C[b][i]
        if F[b, j] < F[a, j]:
            return False
    return True

cdef bool c_is_equal(double[:,:] F, int a, int b):
    cdef int i
    cdef int n_obj = F.shape[1]
    for i in range(n_obj):
        if F[a, i] != F[b, i]:
            return False
    return True
