# distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector

def fast_non_dominated_sort(double[:,:] F, double epsilon = 0.0, n_stop_if_ranked=None):
    return c_fast_non_dominated_sort(F, epsilon, n_stop_if_ranked)

def best_order_sort(double[:,:] F, double epsilon = 0.0, n_stop_if_ranked=None):
    return c_best_order_sort(F, epsilon, n_stop_if_ranked)

def get_relation(F, a, b, epsilon = 0.0):
    return c_get_relation(F, a, b, epsilon)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef c_fast_non_dominated_sort(double[:,:] F, double epsilon = 0.0, n_stop_if_ranked=None):

    cdef:
        int n, i, j, rel, n_ranked
        int[:] ranked, n_dominated
        vector[int] current_front, next_front
        vector[vector[int]] fronts, is_dominating

    # calculate the dominance matrix
    n = F.shape[0]

    fronts = vector[vector[int]]()

    if n == 0:
        return fronts

    # final rank that will be returned
    n_ranked = 0
    ranked = np.zeros(n, dtype=np.intc)

    # for each individual a list of all individuals that are dominated by this one
    is_dominating = vector[vector[int]](n)
    for _ in range(n):
        is_dominating.push_back(vector[int](n))

    # storage for the number of solutions dominated this one
    n_dominated = np.zeros(n, dtype=np.intc)

    current_front = vector[int]()

    for i in range(n):

        for j in range(i + 1, n):

            rel = c_get_relation(F, i, j, epsilon)

            if rel == 1:
                is_dominating[i].push_back(j)
                n_dominated[j] += 1

            elif rel == -1:
                is_dominating[j].push_back(i)
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            current_front.push_back(i)
            ranked[i] = 1
            n_ranked += 1

    # append the first front to the current front
    fronts.push_back(current_front)

    # while not all solutions are assigned to a pareto front or we can stop early because of stop criterium
    while n_ranked < n:

        # if we can stop earlier because of custom number of solutions to be ranked
        if n_stop_if_ranked is not None and n_ranked >= n_stop_if_ranked:
            break

        next_front = vector[int]()

        # for each individual in the current front
        for i in current_front:

            # all solutions that are dominated by this individuals
            for j in is_dominating[i]:

                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.push_back(j)
                    ranked[j] = 1
                    n_ranked += 1

        fronts.push_back(next_front)
        current_front = next_front

    return fronts



@cython.boundscheck(False)
@cython.wraparound(False)
cdef vector[vector[int]] c_best_order_sort(double[:,:] F, double epsilon = 0.0, n_stop_if_ranked=None):


    cdef:
        int n_points, n_obj, n_fronts, n_ranked, i, j, s, e, l
        int[:] rank
        int[:,:] Q
        bool is_dominated
        vector[vector[int]] fronts, empty
        vector[vector[vector[int]]] L

    n_points = F.shape[0]
    n_obj = F.shape[1]
    Q = np.argsort(F, axis=0).astype(np.intc)

    rank = np.full(n_points, -1, dtype=np.intc)

    fronts = vector[vector[int]]()
    fronts.push_back(vector[int]())

    L = vector[vector[vector[int]]]()
    for j in range(n_obj):
        empty = vector[vector[int]]()
        empty.push_back(vector[int]())
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

                        is_dominated = c_get_relation(F, s, e, epsilon) == -1

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

                    fronts.push_back(vector[int]())
                    for l in range(n_obj):
                        L[l].push_back(vector[int]())

                L[j][s_rank].push_back(s)
                fronts[s_rank].push_back(s)
                rank[s] = s_rank
                n_ranked += 1

                if n_ranked == n_points or \
                        (n_stop_if_ranked is not None and n_ranked >= n_stop_if_ranked):
                    break

    return fronts


@cython.boundscheck(False)
@cython.wraparound(False)
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


