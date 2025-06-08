# distutils: language = c++
# cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True


import numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector


cdef extern from "math.h":
    cpdef double floor(double x)

cdef extern from "limits.h":
    int INT_MAX


# ---------------------------------------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------------------------------------



def fast_non_dominated_sort(double[:,:] F, double epsilon = 0.0, int n_stop_if_ranked=INT_MAX, int n_fronts=INT_MAX):
    return c_fast_non_dominated_sort(F, epsilon, n_stop_if_ranked, n_fronts)

def find_non_dominated(double[:,:] F, double epsilon = 0.0):
    return c_find_non_dominated(F, epsilon)

def best_order_sort(double[:,:] F):
    return c_best_order_sort(F)

def get_relation(F, a, b):
    return c_get_relation(F, a, b)

def fast_best_order_sort(double[:,:] F):
    return c_fast_best_order_sort(F)

def efficient_non_dominated_sort(double[:,:] F, strategy="sequential"):
    assert (strategy in ["sequential", 'binary']), "Invalid search strategy"
    return c_efficient_non_dominated_sort(F, strategy)

def dominance_degree_non_dominated_sort(double[:, :] F, strategy="efficient"):
    if strategy not in ["fast", "efficient"]:
        raise ValueError("Invalid search strategy")
    return c_dominance_degree_non_dominated_sort(F, strategy)




# ---------------------------------------------------------------------------------------------------------
# Fast Non-Dominated Sort
# ---------------------------------------------------------------------------------------------------------



cdef vector[vector[int]] c_fast_non_dominated_sort(double[:,:] F, double epsilon = 0.0, int n_stop_if_ranked=INT_MAX, int n_fronts=INT_MAX):

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

    # while not all solutions are assigned to a pareto front or we can stop early because of stop criterion
    while (n_ranked < n_points) and (n_ranked < n_stop_if_ranked) and (fronts.size() < n_fronts):

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


# ---------------------------------------------------------------------------------------------------------
# Optimized Find Non-Dominated
# ---------------------------------------------------------------------------------------------------------

cdef vector[int] c_find_non_dominated(double[:,:] F, double epsilon = 0.0):
    """
    Simple and efficient function to find only non-dominated points.
    Uses the existing optimized c_get_relation function.
    """
    cdef:
        int n_points = F.shape[0]
        int i, j
        bool is_dominated
        vector[int] non_dominated_indices
    
    if n_points == 0:
        return non_dominated_indices
    
    # Check each point to see if it's non-dominated
    for i in range(n_points):
        is_dominated = False
        
        # Check if point i is dominated by any other point j
        for j in range(n_points):
            if i != j:
                # Use the existing optimized c_get_relation function
                # c_get_relation returns 1 if j dominates i, -1 if i dominates j, 0 if neither
                if c_get_relation(F, j, i, epsilon) == 1:
                    is_dominated = True
                    break  # Early termination - no need to check other points
        
        # If point i is not dominated by any other point, it's non-dominated
        if not is_dominated:
            non_dominated_indices.push_back(i)
    
    return non_dominated_indices


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





# ---------------------------------------------------------------------------------------------------------
# Dominance Degree Approach Non-dominated Sort
# ---------------------------------------------------------------------------------------------------------

cdef vector[vector[int]] c_dominance_degree_non_dominated_sort(double[:, :] F, str strategy):
    if strategy == "efficient":
        # return c_dda_ens_get_fronts(c_construct_domination_matrix(F), F.shape[0], np.lexsort(F))
        return c_dda_ens_get_fronts(c_construct_domination_matrix(F), F.shape[1], np.lexsort(F.T))
    elif strategy == "fast":
        # return c_dda_ns_get_fronts(c_construct_domination_matrix(F), F.shape[1], F.shape[0])
        return c_dda_ns_get_fronts(c_construct_domination_matrix(F), F.shape[0], F.shape[1])




# ---------------------------------------------------------------------------------------------------------
# Efficient Non-dominated Sort
# ---------------------------------------------------------------------------------------------------------


cdef vector[vector[int]] c_efficient_non_dominated_sort(double[:,:] F, str strategy):
    cdef:
        long unsigned int i, j, k, n, val
        vector[int] empty, e
        vector[vector[int]] fronts, ret

    # number of individuals
    n = len(F)

    # sort the input lexicographically
    indices = np.lexsort(F.T[::-1])
    F = np.asarray(F)[indices]

    # the fronts to be set for each iteration
    fronts = vector[vector[int]]()

    for i in range(n):

        if strategy == "sequential":
            k = sequential_search(F, i, fronts)
        else:
            k = binary_search(F, i, fronts)

        if k >= fronts.size():
            empty = vector[int]()
            fronts.push_back(empty)

        fronts[k].push_back(i)

    # convert to the return array
    ret = vector[vector[int]]()
    for i in range(fronts.size()):
        e = vector[int]()
        for j in range(fronts[i].size()):
            k = fronts[i][j]
            val = indices[k]
            e.push_back(val)
        ret.push_back(e)

    return ret



cdef int sequential_search(double[:,:] F, int i, vector[vector[int]] fronts):

    cdef:
        int k, j, n_fronts
        bool non_dominated

    n_fronts = fronts.size()
    if n_fronts == 0:
        return 0

    k = 0
    while True:

        non_dominated = True

        # solutions in the k-th front, examine in reverse order
        j = fronts[k].size() - 1

        while j >= 0:
            relation = c_get_relation(F, i, fronts[k][j])
            if relation == -1:
                non_dominated = False
                break
            j = j - 1

        if non_dominated:
            return k

        # move the individual to a new front
        else:
            k += 1
            if k >= n_fronts:
                return n_fronts


cdef int binary_search(double[:,:] F, int i, vector[vector[int]] fronts):

    cdef:
        int n_fronts, k, k_min, k_max, j
        bool non_dominated

    n_fronts = fronts.size()
    if n_fronts == 0:
        return 0

    k_min = 0  # the lower bound for checking
    k_max = n_fronts  # the upper bound for checking
    k = int(floor((k_max + k_min) / 2.0 + 0.5))  # the front now checked

    while True:

        non_dominated = True

        # solutions in the k-th front, examine in reverse order
        j = fronts[k-1].size() - 1

        while j >= 0:
            relation = c_get_relation(F, i, fronts[k-1][j])
            if relation == -1:
                non_dominated = False
                break
            j = j - 1

        # binary search
        if non_dominated:
            if k == k_min + 1:
                return k - 1
            else:
                k_max = k
                k = int(floor((k_max + k_min) / 2.0 + 0.5))

        else:
            k_min = k
            if k_max == k_min + 1 and k_max < n_fronts:
                return k_max - 1
            elif k_min == n_fronts:
                return n_fronts
            else:
                k = int(floor((k_max + k_min) / 2.0 + 0.5))






# ---------------------------------------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------------------------------------


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
    cdef unsigned int i, j
    for i in range(k, C[0].size()):
        j = C[b][i]
        if F[b, j] < F[a, j]:
            return False
    return True

cdef bool c_is_equal(double[:,:] F, int a, int b):
    cdef int i, n_obj = F.shape[1]
    for i in range(n_obj):
        if F[a, i] != F[b, i]:
            return False
    return True

cdef vector[vector[int]] c_construct_domination_matrix(double[:, :]& F):
    cdef:
        long i
        int n = F.shape[0]
        int m = F.shape[1]
        long [:, ::1] b = np.apply_over_axes(np.argsort, F.T, axes=1)

        vector[vector[int]] C = vector[vector[int]](n, vector[int](n, 0))
        vector[vector[int]] D = vector[vector[int]](n, vector[int](n, 0))

    for i in range(m):
        c_construct_comparison_matrix(F[:, i], b[i], C, D, n)

    c_remove_dominators(D, n, m)
    return D

cdef void c_construct_comparison_matrix(double[:]& v, long[:]& b, vector[vector[int]] &C, vector[vector[int]]& D, int n):
    cdef:
        int i, j

    for i in range(n):
        C[b[0]][i] = 1
    for i in range(1, n):
        if v[b[i]] == v[b[i - 1]]:
            for j in range(n):
                C[b[i]][j] = C[b[i - 1]][j]
        else:
            for j in range(i, n):
                C[b[i]][b[j]] = 1

    # increment the DD matrix while also resetting the comparison matrix
    for i in range(n):
        for j in range(n):
            D[i][j] += C[i][j]
            C[i][j] = 0

cdef void c_remove_dominators(vector[vector[int]] &D, int n, int m):
    cdef int i, j
    for i in range(n):
        for j in range(i, n):
            if D[i][j] == m:
                # only perform the row-wise check if the column-wise check fails (C=row major)
                if D[j][i] == m:
                    D[j][i] = 0
                    D[i][j] = 0

cdef void c_remove_front_members(vector[vector[int]] &D, vector[int]& front, int n):
    cdef:
        int i, j

    for i in front:
        for j in range(n):
            # set to -1 so not-yet-added members are preferred by max()
            D[i][j] = -1
            D[j][i] = -1

cdef void c_dda_ns_build_front(vector[int]& max_D, vector[int]& front, int n, int m):
    cdef int i = 0, md
    for md in max_D:
        if 0 <= md < m:
            front.push_back(i)
        i += 1

cdef void c_max(vector[vector[int]]& D, vector[int]& vec_max, int n):
    cdef int i, j, m
    for i in range(n):
        m = -1
        for j in range(n):
            m = max(m, D[j][i])
        vec_max[i] = m

cdef vector[vector[int]] c_dda_ns_get_fronts(vector[vector[int]]& D, int n, int m):
    cdef:
        vector[vector[int]] fronts = vector[vector[int]]()
        vector[int] vec_max = vector[int](n)
        long count = 0

    while count < n:
        front = vector[int]()
        c_max(D, vec_max, n)
        c_dda_ns_build_front(vec_max, front, n, m)
        c_remove_front_members(D, front, n)
        fronts.push_back(front)
        count += front.size()
    return fronts

cdef vector[vector[int]] c_dda_ens_get_fronts(vector[vector[int]]& D, int m, long[::1]& sorted_indices):
    cdef:
        int k, sd, s, n_fronts = 0
        vector[int] fk
        vector[vector[int]] fronts

    for s in range(sorted_indices.shape[0]):
        isinserted = False
        k = 0
        for fk in fronts:
            isdominated = False
            for sd in fk:
                if D[sd][sorted_indices[s]] == m:
                    isdominated = True
                    break
            if not isdominated:
                fronts[k].push_back(sorted_indices[s])
                isinserted = True
                break
            k+= 1
        if not isinserted:
            n_fronts += 1
            fronts.push_back(vector[int](1, sorted_indices[s]))
    return fronts
