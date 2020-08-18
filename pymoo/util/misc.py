from datetime import datetime
from itertools import combinations

import numpy as np
import scipy
import scipy.spatial

from pymoo.model.population import Population
from pymoo.model.sampling import Sampling


def parameter_less(F, CV):
    ret = np.copy(F)
    parameter_less = np.max(F, axis=0) + CV

    infeasible = CV > 0
    ret[infeasible] = parameter_less[infeasible]

    return ret


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
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def norm_euclidean_distance(problem):
    return lambda a, b: np.sqrt((((a - b) / (problem.xu - problem.xl)) ** 2).sum(axis=1))


def cdist(A, B, **kwargs):
    return scipy.spatial.distance.cdist(A, B, **kwargs)


def vectorized_cdist(A, B, func_dist=euclidean_distance, fill_diag_with_inf=False, **kwargs):
    u = np.repeat(A, B.shape[0], axis=0)
    v = np.tile(B, (A.shape[0], 1))

    D = func_dist(u, v, **kwargs)
    M = np.reshape(D, (A.shape[0], B.shape[0]))

    if fill_diag_with_inf:
        np.fill_diagonal(M, np.inf)

    return M


def norm_eucl_dist(problem, A, B, **kwargs):
    return vectorized_cdist(A, B, func_dist=norm_euclidean_distance(problem), **kwargs)


def covert_to_type(problem, X):
    if problem.type_var == np.double:
        return X.astype(np.double)
    elif problem.type_var == np.int:
        return np.round(X).astype(np.int)
    elif problem.type_var == np.bool:
        return X < (problem.xu - problem.xl) / 2


def find_duplicates(X, epsilon=1e-16):
    # calculate the distance matrix from each point to another
    D = cdist(X, X)

    # set the diagonal to infinity
    D[np.triu_indices(len(X))] = np.inf

    # set as duplicate if a point is really close to this one
    is_duplicate = np.any(D < epsilon, axis=1)

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


def all_except(x, *args):
    if len(args) == 0:
        return x
    else:
        H = set(args) if len(args) > 5 else args
        I = [k for k in range(len(x)) if k not in H]
        return x[I]


def all_combinations(A, B):
    u = np.repeat(A, B.shape[0], axis=0)
    v = np.tile(B, A.shape[0])
    return np.column_stack([u, v])


def pop_from_sampling(problem, sampling, n_initial_samples, pop=None):
    # the population type can be different - (different type of individuals)
    if pop is None:
        pop = Population()

    # provide a whole population object - (individuals might be already evaluated)
    if isinstance(sampling, Population):
        pop = sampling

    else:
        # if just an X array create a pop
        if isinstance(sampling, np.ndarray):
            pop = pop.new("X", sampling)

        elif isinstance(sampling, Sampling):
            # use the sampling
            pop = sampling.do(problem, n_initial_samples, pop=pop)

        else:
            return None

    return pop


def evaluate_if_not_done_yet(evaluator, problem, pop, algorithm=None):
    I = np.where(pop.get("F") == None)[0]
    if len(I) > 0:
        pop[I] = evaluator.process(problem, pop[I], algorithm=algorithm)


def set_if_none(kwargs, str, val):
    if str not in kwargs:
        kwargs[str] = val


def set_if_none_from_tuples(kwargs, *args):
    for key, val in args:
        if key not in kwargs:
            kwargs[key] = val


def calc_perpendicular_distance(N, ref_dirs):
    u = np.tile(ref_dirs, (len(N), 1))
    v = np.repeat(N, len(ref_dirs), axis=0)

    norm_u = np.linalg.norm(u, axis=1)

    scalar_proj = np.sum(v * u, axis=1) / norm_u
    proj = scalar_proj[:, None] * u / norm_u[:, None]
    val = np.linalg.norm(proj - v, axis=1)
    matrix = np.reshape(val, (len(N), len(ref_dirs)))

    return matrix


def distance_of_closest_points_to_others(X):
    D = vectorized_cdist(X, X)
    np.fill_diagonal(D, np.inf)
    return D.argmin(axis=1), D.min(axis=1)


def time_to_int(t):
    td = datetime.strptime(t, '%H:%M:%S') - datetime(1900, 1, 1)
    return td.total_seconds()


def powerset(iterable):
    for n in range(len(iterable) + 1):
        yield from combinations(iterable, n)


def intersect(a, b):
    H = set()
    for entry in b:
        H.add(entry)

    ret = []
    for entry in a:
        if entry in H:
            ret.append(entry)

    return ret


def has_feasible(pop):
    return np.any(pop.get("feasible"))


def to_numpy(a):
    return np.array(a)


def termination_from_tuple(termination):
    from pymoo.model.termination import Termination

    # get the termination if provided as a tuple - create an object
    if termination is not None and not isinstance(termination, Termination):
        from pymoo.factory import get_termination
        if isinstance(termination, str):
            termination = get_termination(termination)
        else:
            termination = get_termination(*termination)

    return termination
