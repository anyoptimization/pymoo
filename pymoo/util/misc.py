import numpy as np
import scipy


def swap(M, a, b):
    tmp = M[a]
    M[a] = M[b]
    M[b] = tmp


# repairs a numpy array to be in bounds
def repair(X, xl, xu):
    larger_than_xu = X[0, :] > xu
    X[0, larger_than_xu] = xu[larger_than_xu]

    smaller_than_xl = X[0, :] < xl
    X[0, smaller_than_xl] = xl[smaller_than_xl]

    return X


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def parameter_less_constraints(F, CV, F_max=None):
    if F_max is None:
        F_max = np.max(F)
    has_constraint_violation = CV > 0
    F[has_constraint_violation] = CV[has_constraint_violation] + F_max
    return F


def random_permuations(n, l):
    from pymoo.rand import random
    perms = []
    for i in range(n):
        perms.append(random.perm(l))
    P = np.concatenate(perms)
    return P


def get_duplicates(M):
    res = []
    I = np.lexsort([M[:, i] for i in reversed(range(0, M.shape[1]))])
    S = M[I, :]

    i = 0

    while i < S.shape[0] - 1:
        l = []
        while np.all(S[i, :] == S[i + 1, :]):
            l.append(I[i])
            i += 1
        if len(l) > 0:
            l.append(I[i])
            res.append(l)
        i += 1

    return res


def cdist(A, B, **kwargs):
    if A.dtype != np.object:
        return scipy.spatial.distance.cdist(A, B, **kwargs)
    else:
        D = np.full((A.shape[0], B.shape[1]), np.inf, dtype=np.float)
        for i in range(A.shape[0]):
            for j in range(i + 1, B.shape[1]):
                d = M[i].distance_to(M[j])
                D[i, j], D[j, i] = d, d
        return D


def vectorized_cdist(A, B, func_dist):
    u = np.repeat(A, B.shape[0], axis=0)
    v = np.tile(B, (A.shape[0], 1))

    D = func_dist(u, v)
    M = np.reshape(D, (A.shape[0], B.shape[0]))
    return M


def covert_to_type(problem, X):
    if problem.type_var == np.double:
        return X.astype(np.double)
    elif problem.type_var == np.int:
        return np.round(X).astype(np.int)
    elif problem.type_var == np.bool:
        return X < (problem.xu - problem.xl) / 2


if __name__ == '__main__':
    M = np.random.random((100, 3))

    M[3, :] = M[55, :]
    M[10, :] = M[55, :]

    print(get_duplicates(M))
