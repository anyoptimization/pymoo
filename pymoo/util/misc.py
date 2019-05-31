import numpy as np
import scipy
import scipy.spatial


def parameter_less(F, CV):
    val = np.copy(F)
    parameter_less = np.max(F, axis=0) + CV

    infeasible = CV > 0
    val[infeasible] = parameter_less[infeasible]

    return val


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
    perms = []
    for i in range(n):
        perms.append(np.random.permutation(l))
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


def euclidean_distance(a, b):
    return np.sqrt((a - b) ** 2)


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


def vectorized_cdist(A, B, func_dist=euclidean_distance):
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


def filter_duplicate(X, epsilon=1e-16):
    # value to finally return
    is_duplicate = np.full(len(X), False)

    # check for duplicates in pop itself
    D = cdist(X, X)
    D[np.triu_indices(len(X))] = np.inf
    is_duplicate = np.logical_or(is_duplicate, np.any(D < epsilon, axis=1))

    return is_duplicate


def at_least_2d_array(x, extend_as="row"):
    if not isinstance(x, np.ndarray):
        x = np.array([x])

    if x.ndim == 1:
        if extend_as == "row":
            x = x[None, :]
        elif extend_as == "column":
            x = x[:, None]

    return x


def to_1d_array_if_possible(x):
    if not isinstance(x, np.ndarray):
        x = np.array([x])

    if x.ndim == 2:
        if x.shape[0] == 1 or x.shape[1] == 1:
            x = x.flatten()

    return x


def stack(*args, flatten=True):
    if not flatten:
        ps = np.concatenate([e[None, ...] for e in args])
    else:
        ps = np.row_stack(args)
    return ps


def all_combinations(A, B):
    u = np.repeat(A, B.shape[0], axis=0)
    v = np.tile(B, A.shape[0])
    return np.column_stack([u,v])

if __name__ == '__main__':
    x = np.linspace(0, 5, 50)
    X = all_combinations(x, x)

    M = np.random.random((100, 3))

    M[3, :] = M[55, :]
    M[10, :] = M[55, :]

    print(get_duplicates(M))
