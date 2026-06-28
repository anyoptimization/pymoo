"""Miscellaneous utility functions for optimization."""

import typing
from collections import OrderedDict
from itertools import combinations
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from pymoo.core.population import Population
from pymoo.core.sampling import Sampling
from pymoo.util import default_random_state

if typing.TYPE_CHECKING:
    from pymoo.core.termination import Termination


def parameter_less(F: Any, CV: Any, fmax: Optional[Any] = None, inplace: bool = False) -> Any:
    """Apply parameter-less penalty method to infeasible solutions.

    Args:
        F: Objective function values.
        CV: Constraint violations.
        fmax: Maximum objective value (if None, computed from F).
        inplace: Whether to modify F in place.

    Returns:
        Modified objective values with penalties applied.
    """
    assert len(F) == len(CV)

    if not inplace:
        F = np.copy(F)

    if fmax is None:
        fmax = np.max(F)

    param_less = fmax + CV

    infeas = (CV > 0).flatten()
    F[infeas] = param_less[infeas]

    return F


def swap(M: Any, a: int, b: int) -> None:
    """Swap two rows in a 2D array in-place.

    Args:
        M: The array.
        a: Index of first row.
        b: Index of second row.
    """
    tmp = M[a]
    M[a] = M[b]
    M[b] = tmp


def repair(X: Any, xl: Any, xu: Any) -> Any:
    """Repair a solution to be within bounds.

    Args:
        X: Solution array.
        xl: Lower bounds.
        xu: Upper bounds.

    Returns:
        Clipped solution.
    """
    X[0] = np.clip(X[0], xl, xu)
    return X


def unique_rows(a: Any) -> Any:
    """Get unique rows from a 2D array.

    Args:
        a: 2D array.

    Returns:
        Array of unique rows.
    """
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([("", a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def parameter_less_constraints(F: Any, CV: Any, F_max: Optional[Any] = None) -> Any:
    """Apply parameter-less penalty to constraint-violated solutions.

    Args:
        F: Objective function values.
        CV: Constraint violations.
        F_max: Maximum objective value.

    Returns:
        Modified objectives.
    """
    if F_max is None:
        F_max = np.max(F)
    has_constraint_violation = CV > 0
    F[has_constraint_violation] = CV[has_constraint_violation] + F_max
    return F


@default_random_state
def random_permutations(
    n: int,
    l: int,  # noqa: E741
    concat: bool = True,
    random_state: Any = None,
) -> Any:
    """Generate random permutations.

    Args:
        n: Number of permutations.
        l: Length of each permutation.
        concat: Whether to concatenate results.
        random_state: Random state generator.

    Returns:
        Random permutations as array.
    """
    P: Any = []
    for i in range(n):
        P.append(random_state.permutation(l))
    if concat:
        P = np.concatenate(P)
    return P


def get_duplicates(M: Any) -> list:
    """Find duplicate rows in a 2D array.

    Args:
        M: 2D array.

    Returns:
        List of lists containing indices of duplicate rows.
    """
    res = []
    I = np.lexsort([M[:, i] for i in reversed(range(0, M.shape[1]))])  # noqa: E741
    S = M[I, :]

    i = 0

    while i < S.shape[0] - 1:
        l = []  # noqa: E741
        while np.all(S[i, :] == S[i + 1, :]):
            l.append(I[i])
            i += 1
        if len(l) > 0:
            l.append(I[i])
            res.append(l)
        i += 1

    return res


def func_euclidean_distance(a: Any, b: Any) -> Any:
    """Compute euclidean distance between rows.

    Args:
        a: First array.
        b: Second array.

    Returns:
        Euclidean distances.
    """
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def func_norm_euclidean_distance(xl: Any, xu: Any) -> Callable[[Any, Any], Any]:
    """Create normalized euclidean distance function.

    Args:
        xl: Lower bounds.
        xu: Upper bounds.

    Returns:
        Distance function.
    """
    return lambda a, b: np.sqrt((((a - b) / (xu - xl)) ** 2).sum(axis=1))


def norm_eucl_dist_by_bounds(A: Any, B: Any, xl: Any, xu: Any, **kwargs: Any) -> Any:
    """Compute normalized euclidean distance by bounds.

    Args:
        A: First population.
        B: Second population.
        xl: Lower bounds.
        xu: Upper bounds.
        **kwargs: Additional arguments.

    Returns:
        Distance matrix.
    """
    return vectorized_cdist(A, B, func_dist=func_norm_euclidean_distance(xl, xu), **kwargs)


def norm_eucl_dist(problem: Any, A: Any, B: Any, **kwargs: Any) -> Any:
    """Compute normalized euclidean distance using problem bounds.

    Args:
        problem: Optimization problem.
        A: First population.
        B: Second population.
        **kwargs: Additional arguments.

    Returns:
        Distance matrix.
    """
    return norm_eucl_dist_by_bounds(A, B, *problem.bounds(), **kwargs)


def func_manhatten_distance(a: Any, b: Any) -> Any:
    """Compute manhattan distance between rows.

    Args:
        a: First array.
        b: Second array.

    Returns:
        Manhattan distances.
    """
    return np.abs(a - b).sum(axis=1)


def func_norm_manhatten_distance(xl: Any, xu: Any) -> Callable[[Any, Any], Any]:
    """Create normalized manhattan distance function.

    Args:
        xl: Lower bounds.
        xu: Upper bounds.

    Returns:
        Distance function.
    """
    return lambda a, b: np.abs((a - b) / (xu - xl)).sum(axis=1)


def norm_manhatten_dist_by_bounds(A: Any, B: Any, xl: Any, xu: Any, **kwargs: Any) -> Any:
    """Compute normalized manhattan distance by bounds.

    Args:
        A: First population.
        B: Second population.
        xl: Lower bounds.
        xu: Upper bounds.
        **kwargs: Additional arguments.

    Returns:
        Distance matrix.
    """
    return vectorized_cdist(A, B, func_dist=func_norm_manhatten_distance(xl, xu), **kwargs)


def norm_manhatten_dist(problem: Any, A: Any, B: Any, **kwargs: Any) -> Any:
    """Compute normalized manhattan distance using problem bounds.

    Args:
        problem: Optimization problem.
        A: First population.
        B: Second population.
        **kwargs: Additional arguments.

    Returns:
        Distance matrix.
    """
    return norm_manhatten_dist_by_bounds(A, B, *problem.bounds(), **kwargs)


def func_tchebychev_distance(a: Any, b: Any) -> Any:
    """Compute Chebyshev distance between rows.

    Args:
        a: First array.
        b: Second array.

    Returns:
        Chebyshev distances.
    """
    return np.abs(a - b).max(axis=1)


def func_norm_tchebychev_distance(xl: Any, xu: Any) -> Callable[[Any, Any], Any]:
    """Create normalized Chebyshev distance function.

    Args:
        xl: Lower bounds.
        xu: Upper bounds.

    Returns:
        Distance function.
    """
    return lambda a, b: np.abs((a - b) / (xu - xl)).max(axis=1)


def norm_tchebychev_dist_by_bounds(A: Any, B: Any, xl: Any, xu: Any, **kwargs: Any) -> Any:
    """Compute normalized Chebyshev distance by bounds.

    Args:
        A: First population.
        B: Second population.
        xl: Lower bounds.
        xu: Upper bounds.
        **kwargs: Additional arguments.

    Returns:
        Distance matrix.
    """
    return vectorized_cdist(A, B, func_dist=func_norm_tchebychev_distance(xl, xu), **kwargs)


def norm_tchebychev_dist(problem: Any, A: Any, B: Any, **kwargs: Any) -> Any:
    """Compute normalized Chebyshev distance using problem bounds.

    Args:
        problem: Optimization problem.
        A: First population.
        B: Second population.
        **kwargs: Additional arguments.

    Returns:
        Distance matrix.
    """
    return norm_tchebychev_dist_by_bounds(A, B, *problem.bounds(), **kwargs)


def cdist(A: Any, B: Any, **kwargs: Any) -> Any:
    """Compute pairwise distances using scipy.

    Args:
        A: First array.
        B: Second array.
        **kwargs: Additional arguments to scipy.spatial.distance.cdist.

    Returns:
        Distance matrix.
    """
    from scipy.spatial import distance

    return distance.cdist(A.astype(float), B.astype(float), **kwargs)


def vectorized_cdist(
    A: Any,
    B: Any,
    func_dist: Callable[[Any, Any], Any] = func_euclidean_distance,
    fill_diag_with_inf: bool = False,
    **kwargs: Any,
) -> Any:
    """Compute pairwise distances using a custom distance function.

    Args:
        A: First array.
        B: Second array.
        func_dist: Distance function.
        fill_diag_with_inf: Whether to fill diagonal with infinity.
        **kwargs: Additional arguments.

    Returns:
        Distance matrix.
    """
    assert A.ndim <= 2 and B.ndim <= 2

    A, only_row = at_least_2d_array(A, extend_as="row", return_if_reshaped=True)
    B, only_column = at_least_2d_array(B, extend_as="row", return_if_reshaped=True)

    u = np.repeat(A, B.shape[0], axis=0)
    v = np.tile(B, (A.shape[0], 1))

    D = func_dist(u, v, **kwargs)
    M = np.reshape(D, (A.shape[0], B.shape[0]))

    if fill_diag_with_inf:
        np.fill_diagonal(M, np.inf)

    if only_row and only_column:
        M = M[0, 0]
    elif only_row:
        M = M[0]
    elif only_column:
        M = M[:, [0]]

    return M


def covert_to_type(problem: Any, X: Any) -> Any:
    """Convert solution array to problem variable type.

    Args:
        problem: Optimization problem.
        X: Solution array.

    Returns:
        Converted solution array.
    """
    if problem.vtype is float:  # noqa: E721
        return X.astype(np.double)
    elif problem.vtype is int:  # noqa: E721
        return np.round(X).astype(int)
    elif problem.vtype is bool:  # noqa: E721
        return X < (problem.xu - problem.xl) / 2


def find_duplicates(X: Any, epsilon: float = 1e-16) -> Any:
    """Find duplicate solutions in population.

    Args:
        X: Solution array.
        epsilon: Tolerance for duplicate detection.

    Returns:
        Boolean array indicating duplicates.
    """
    D = cdist(X, X)

    D[np.triu_indices(len(X))] = np.inf

    is_duplicate = np.any(D <= epsilon, axis=1)

    return is_duplicate


def at_least_2d(*args: Any, **kwargs: Any) -> Any:
    """Ensure all arguments are at least 2D arrays.

    Args:
        *args: Arrays to reshape.
        **kwargs: Additional arguments.

    Returns:
        Reshaped array or tuple of arrays.
    """
    ret = tuple([at_least_2d_array(arg, **kwargs) for arg in args])
    if len(ret) == 1:
        ret = ret[0]
    return ret


def at_least_2d_array(x: Any, extend_as: str = "row", return_if_reshaped: bool = False) -> Union[Any, Tuple[Any, bool]]:
    """Ensure array is at least 2D.

    Args:
        x: Input array.
        extend_as: How to extend 1D arrays ('row' or 'column').
        return_if_reshaped: Whether to return reshape flag.

    Returns:
        2D array, optionally with reshape flag.
    """
    if x is None:
        return x
    elif not isinstance(x, np.ndarray):
        x = np.array([x])

    has_been_reshaped = False

    if x.ndim == 1:
        if extend_as.startswith("r"):
            x = x[None, :]
        elif extend_as.startswith("c"):
            x = x[:, None]
        else:
            raise Exception("The option `extend_as` should be either `row` or `column`.")

        has_been_reshaped = True

    if return_if_reshaped:
        return x, has_been_reshaped
    else:
        return x


def to_1d_array_if_possible(x: Any) -> Any:
    """Convert array to 1D if possible.

    Args:
        x: Input array.

    Returns:
        1D array if applicable, otherwise original shape.
    """
    if not isinstance(x, np.ndarray):
        x = np.array([x])

    if x.ndim == 2:
        if x.shape[0] == 1 or x.shape[1] == 1:
            x = x.flatten()

    return x


def stack(*args: Any, flatten: bool = True) -> Any:
    """Stack arrays along axis.

    Args:
        *args: Arrays to stack.
        flatten: Whether to stack as rows or add new axis.

    Returns:
        Stacked array.
    """
    if not flatten:
        ps = np.concatenate([e[None, ...] for e in args])
    else:
        ps = np.vstack(args)
    return ps


def all_except(x: Any, *args: Any) -> Any:
    """Get all indices of x except those in args.

    Args:
        x: Sequence.
        *args: Indices to exclude.

    Returns:
        Filtered sequence.
    """
    if len(args) == 0:
        return x
    else:
        H = set(args) if len(args) > 5 else args
        I = [k for k in range(len(x)) if k not in H]  # noqa: E741
        return x[I]


def all_combinations(A: Any, B: Any) -> Any:
    """Get all combinations of rows from A and B.

    Args:
        A: First array.
        B: Second array.

    Returns:
        Array of all combinations.
    """
    u = np.repeat(A, B.shape[0], axis=0)
    v = np.tile(B, A.shape[0])
    return np.column_stack([u, v])


def pop_from_sampling(
    problem: Any,
    sampling: Any,
    n_initial_samples: int,
    pop: Optional[Population] = None,
) -> Optional[Population]:
    """Create population from sampling.

    Args:
        problem: Optimization problem.
        sampling: Sampling strategy or population.
        n_initial_samples: Number of initial samples.
        pop: Initial population.

    Returns:
        Population.
    """
    if pop is None:
        pop = Population()

    if isinstance(sampling, Population):
        pop = sampling

    else:
        if isinstance(sampling, np.ndarray):
            pop = pop.new("X", sampling)

        elif isinstance(sampling, Sampling):
            pop = sampling.do(problem, n_initial_samples, pop=pop)

        else:
            return None

    return pop


def evaluate_if_not_done_yet(evaluator: Any, problem: Any, pop: Any, algorithm: Optional[Any] = None) -> None:
    """Evaluate individuals that haven't been evaluated yet.

    Args:
        evaluator: Evaluator instance.
        problem: Optimization problem.
        pop: Population.
        algorithm: Algorithm instance.
    """
    I = np.where(pop.get("F") is None)[0]  # noqa: E741
    if len(I) > 0:
        pop[I] = evaluator.process(problem, pop[I], algorithm=algorithm)


def set_if_none(kwargs: dict, str: str, val: Any) -> None:
    """Set dictionary value if key is not present.

    Args:
        kwargs: Dictionary.
        str: Key.
        val: Value to set.
    """
    if str not in kwargs:
        kwargs[str] = val


def set_if_none_from_tuples(kwargs: dict, *args: Tuple[str, Any]) -> None:
    """Set dictionary values from tuples if keys are not present.

    Args:
        kwargs: Dictionary.
        *args: Key-value tuples.
    """
    for key, val in args:
        if key not in kwargs:
            kwargs[key] = val


def distance_of_closest_points_to_others(X: Any) -> Tuple[Any, Any]:
    """Find closest neighbor to each point.

    Args:
        X: Point array.

    Returns:
        Indices and distances of closest neighbors.
    """
    D = vectorized_cdist(X, X)
    np.fill_diagonal(D, np.inf)
    return D.argmin(axis=1), D.min(axis=1)


def time_to_int(t: str) -> int:
    """Convert time string to seconds.

    Args:
        t: Time string in format HH:MM:SS.

    Returns:
        Time in seconds.
    """
    vals = [int(e) for e in t.split(":")][::-1]
    s = vals[0]
    if len(vals) > 1:
        s += 60 * vals[1]
    if len(vals) > 2:
        s += 3600 * vals[2]
    return s


def powerset(iterable: Any) -> Any:
    """Generate all subsets of an iterable.

    Args:
        iterable: Input iterable.

    Yields:
        All subsets.
    """
    for n in range(len(iterable) + 1):
        yield from combinations(iterable, n)


def intersect(a: Any, b: Any) -> list:
    """Find intersection of two iterables.

    Args:
        a: First iterable.
        b: Second iterable.

    Returns:
        Elements in both a and b.
    """
    H = set()
    for entry in b:
        H.add(entry)

    ret = []
    for entry in a:
        if entry in H:
            ret.append(entry)

    return ret


def has_feasible(pop: Any) -> bool:
    """Check if population has feasible solutions.

    Args:
        pop: Population.

    Returns:
        True if any feasible solutions exist.
    """
    return bool(np.any(pop.get("FEAS")))


def to_numpy(a: Any) -> Any:
    """Convert to numpy array.

    Args:
        a: Input array-like.

    Returns:
        Numpy array.
    """
    return np.array(a)


def termination_from_tuple(
    termination: Union["Termination", str, Tuple[str, ...]],
) -> Optional["Termination"]:
    """Create termination object from tuple or string.

    Args:
        termination: Termination specification.

    Returns:
        Termination object.
    """
    from pymoo.core.termination import Termination

    if termination is not None and not isinstance(termination, Termination):
        from pymoo.termination import get_termination

        if isinstance(termination, str):
            termination = get_termination(termination)
        else:
            termination = get_termination(*termination)

    return termination


def unique_and_all_indices(arr: Any) -> Tuple[Any, list]:
    """Get unique values and all indices grouped by value.

    Args:
        arr: Input array.

    Returns:
        Unique values and list of index arrays.
    """
    sort_indexes = np.argsort(arr)
    arr = np.asarray(arr)[sort_indexes]
    vals, first_indexes, inverse, counts = np.unique(arr, return_index=True, return_inverse=True, return_counts=True)
    indexes = np.split(sort_indexes, first_indexes[1:])
    for x in indexes:
        x.sort()
    return vals, indexes


def from_dict(D: dict, *keys: str) -> list:
    """Extract values from dictionary.

    Args:
        D: Dictionary.
        *keys: Keys to extract.

    Returns:
        List of values.
    """
    return [D.get(k) for k in keys]


def list_of_dicts_unique(l: list, k: str) -> list:  # noqa: E741
    """Get unique elements from list of dicts by key.

    Args:
        l: List of dictionaries.
        k: Key to use for uniqueness.

    Returns:
        List of unique dictionaries.
    """
    return list(OrderedDict([(e[k], None) for e in l]).keys())


def list_of_dicts_filter(l: list, *pairs: Tuple[str, Any]) -> list:  # noqa: E741
    """Filter list of dicts by key-value pairs.

    Args:
        l: List of dictionaries.
        *pairs: Key-value filter pairs.

    Returns:
        Filtered list.
    """
    return [e for e in l if all(e[k] == v for (k, v) in pairs)]


def logical_op(func: Callable[[Any, Any], Any], a: Any, b: Any, *args: Any) -> Any:
    """Apply logical operation across multiple arguments.

    Args:
        func: Binary function to apply.
        a: First argument.
        b: Second argument.
        *args: Additional arguments.

    Returns:
        Result of operation.
    """
    ret = func(a, b)
    for c in args:
        ret = func(ret, c)
    return ret


def replace_nan_by(x: Any, val: Any, inplace: bool = False) -> Any:
    """Replace NaN values in array.

    Args:
        x: Input array.
        val: Replacement value.
        inplace: Whether to modify in place.

    Returns:
        Array with NaNs replaced.
    """
    is_nan = np.isnan(x)
    if np.sum(is_nan) > 0:
        if not inplace:
            x = x.copy()
        x[is_nan] = val
    return x


def set_defaults(
    kwargs: dict,
    defaults: dict,
    overwrite: bool = False,
    func_get: Callable[[Any], Any] = lambda x: x,
) -> None:
    """Set default values in dictionary.

    Args:
        kwargs: Target dictionary.
        defaults: Default values to set.
        overwrite: Whether to overwrite existing keys.
        func_get: Function to transform default values.
    """
    for k, v in defaults.items():
        if overwrite or k not in kwargs:
            kwargs[k] = func_get(v)


def filter_params(params: dict, prefix: str, delete_prefix: bool = True) -> dict:
    """Filter dictionary keys by prefix.

    Args:
        params: Dictionary to filter.
        prefix: Key prefix to match.
        delete_prefix: Whether to remove prefix from keys.

    Returns:
        Filtered dictionary.
    """
    ret = {}
    for k, v in params.items():
        if k.startswith(prefix):
            if delete_prefix:
                k = k[len(prefix) :]
            ret[k] = v
    return ret


def where_is_what(x: Any) -> dict:
    """Group indices by element value.

    Args:
        x: Input iterable.

    Returns:
        Dictionary mapping values to indices.
    """
    H: dict[Any, list] = {}
    for k, e in enumerate(x):
        if e not in H:
            H[e] = []
        H[e].append(k)
    return H


def crossover_mask(X: Any, M: Any) -> Any:
    """Apply crossover mask to solution pair.

    Args:
        X: Solution pair array.
        M: Crossover mask.

    Returns:
        Crossed over solutions.
    """
    _X = np.copy(X)
    _X[0][M] = X[1][M]
    _X[1][M] = X[0][M]
    return _X


@default_random_state
def row_at_least_once_true(M: Any, random_state: Any = None) -> Any:
    """Ensure each row has at least one True value.

    Args:
        M: Boolean matrix.
        random_state: Random state generator.

    Returns:
        Modified matrix with all rows having at least one True.
    """
    _, d = M.shape
    for k in np.where(~np.any(M, axis=1))[0]:
        M[k, random_state.integers(d)] = True
    return M
