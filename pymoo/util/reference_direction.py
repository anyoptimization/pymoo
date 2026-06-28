"""Reference direction generation for multi-objective optimization."""

import sys

import numpy as np
from numpy.typing import NDArray
from scipy import special

from pymoo.util.misc import find_duplicates, cdist
from pymoo.util import default_random_state


def default_ref_dirs(m: int) -> NDArray:
    """Get default reference directions for given number of objectives.

    Args:
        m: Number of objectives (1, 2, or 3).

    Returns:
        Array of reference directions.

    Raises:
        Exception: If m > 3.
    """
    if m == 1:
        return np.array([[1.0]])
    elif m == 2:
        return UniformReferenceDirectionFactory(m, n_partitions=99).do()
    elif m == 3:
        return UniformReferenceDirectionFactory(m, n_partitions=12).do()
    else:
        raise Exception(
            "No default reference directions for more than 3 objectives. "
            "Please provide them directly: "
            "https://pymoo.org/misc/reference_directions.html"
        )


class ReferenceDirectionFactory:
    """Base class for reference direction factories."""

    def __init__(
        self,
        n_dim: int,
        scaling: float | None = None,
        lexsort: bool = True,
        verbose: bool = False,
        **kwargs,  # type: ignore[no-untyped-def]
    ) -> None:
        """Initialize the reference direction factory.

        Args:
            n_dim: Number of objectives.
            scaling: Scaling factor for reference directions.
            lexsort: Whether to lexicographically sort the directions.
            verbose: Whether to print verbose output.
            **kwargs: Additional arguments.
        """
        super().__init__()
        self.n_dim = n_dim
        self.scaling = scaling
        self.lexsort = lexsort
        self.verbose = verbose

    def __call__(self) -> NDArray:
        """Generate reference directions.

        Returns:
            Array of reference directions.
        """
        return self.do()

    @default_random_state(seed=1)
    def do(self, random_state=None) -> NDArray:  # type: ignore[misc]
        """Generate reference directions.

        Args:
            random_state: Random state for reproducibility.

        Returns:
            Array of reference directions.
        """
        if self.n_dim == 1:
            return np.array([[1.0]])
        else:
            val = self._do(random_state=random_state)
            if isinstance(val, tuple):
                ref_dirs = val[0]
            else:
                ref_dirs = val

            if ref_dirs is None:  # pragma: no cover
                raise RuntimeError("Reference directions could not be generated")

            if self.scaling is not None:
                ref_dirs = scale_reference_directions(ref_dirs, self.scaling)

            # do ref_dirs is desired
            if self.lexsort:
                I = np.lexsort(  # noqa: E741
                    [ref_dirs[:, j] for j in range(ref_dirs.shape[1])][::-1]
                )
                ref_dirs = ref_dirs[I]

            return ref_dirs

    def _do(self, random_state=None) -> NDArray | None:  # type: ignore[misc]
        """Subclass implementation for generating reference directions.

        Args:
            random_state: Random state for reproducibility.

        Returns:
            Array of reference directions or None.
        """
        return None


def get_number_of_uniform_points(n_partitions: int, n_dim: int) -> int:
    """Get the number of uniform points for given partitions and dimensions.

    Args:
        n_partitions: Number of partitions.
        n_dim: Number of dimensions.

    Returns:
        The number of uniform points that can be created.
    """
    return int(special.binom(n_dim + n_partitions - 1, n_partitions))


def get_partition_closest_to_points(n_points: int, n_dim: int) -> int:
    """Get partition number that creates desired number of points or less.

    Args:
        n_points: Desired number of points.
        n_dim: Number of dimensions.

    Returns:
        The partition number.
    """
    if n_dim == 1:
        return 0

    n_partitions = 1
    _n_points = get_number_of_uniform_points(n_partitions, n_dim)
    while _n_points <= n_points:
        n_partitions += 1
        _n_points = get_number_of_uniform_points(n_partitions, n_dim)
    return n_partitions - 1


def das_dennis(n_partitions: int, n_dim: int) -> NDArray:
    """Generate Das Dennis reference directions.

    Args:
        n_partitions: Number of partitions.
        n_dim: Number of dimensions.

    Returns:
        Array of reference directions.
    """
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs: list[NDArray] = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)


def das_dennis_recursion(
    ref_dirs: list[NDArray],
    ref_dir: NDArray,
    n_partitions: int,
    beta: int,
    depth: int,
) -> None:
    """Recursively generate Das Dennis reference directions.

    Args:
        ref_dirs: List to accumulate reference directions.
        ref_dir: Current reference direction being built.
        n_partitions: Total number of partitions.
        beta: Current beta value for recursion.
        depth: Current depth in recursion.
    """
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


class UniformReferenceDirectionFactory(ReferenceDirectionFactory):
    """Factory for generating uniform reference directions using Das Dennis method."""

    def __init__(
        self,
        n_dim: int,
        scaling: float | None = None,
        n_points: int | None = None,
        n_partitions: int | None = None,
        **kwargs,  # type: ignore[no-untyped-def]
    ) -> None:
        """Initialize uniform reference direction factory.

        Args:
            n_dim: Number of objectives.
            scaling: Scaling factor for reference directions.
            n_points: Number of reference directions to generate (optional).
            n_partitions: Number of partitions (optional).
            **kwargs: Additional arguments for parent class.

        Raises:
            Exception: If both n_points and n_partitions are None, or if n_points
                doesn't match any partition number.
        """
        super().__init__(n_dim, scaling=scaling, **kwargs)

        if n_points is not None:
            n_partitions = get_partition_closest_to_points(n_points, n_dim)
            results_in = get_number_of_uniform_points(n_partitions, n_dim)

            # the number of points are not matching to any partition number
            if results_in != n_points:
                results_in_next = get_number_of_uniform_points(n_partitions + 1, n_dim)
                raise Exception(
                    f"The number of points (n_points = {n_points}) "
                    "can not be created uniformly.\n"
                    f"Either choose n_points = {results_in} "
                    f"(n_partitions = {n_partitions}) or "
                    f"n_points = {results_in_next} "
                    f"(n_partitions = {n_partitions + 1})."
                )

            self.n_partitions = n_partitions

        elif n_partitions is not None:
            self.n_partitions = n_partitions

        else:
            raise Exception("Either provide number of partitions or number of points.")

    def _do(self, random_state=None) -> NDArray:  # type: ignore[misc]
        """Generate uniform reference directions.

        Args:
            random_state: Random state (unused).

        Returns:
            Array of reference directions.
        """
        return das_dennis(self.n_partitions, self.n_dim)


class MultiLayerReferenceDirectionFactory:
    """Factory for combining multiple reference direction factories into layers."""

    def __init__(self, *args) -> None:
        """Initialize with reference direction factories as layers.

        Args:
            *args: Reference direction factories or arrays.
        """
        self.layers: list[NDArray] = []
        self.layers.extend(args)

    def __call__(self) -> NDArray:
        """Generate multi-layer reference directions.

        Returns:
            Array of reference directions.
        """
        return self.do()

    def add_layer(self, *args) -> None:
        """Add additional layers.

        Args:
            *args: Reference direction factories or arrays to add.
        """
        self.layers.extend(args)

    def do(self) -> NDArray:
        """Combine all layers and remove duplicates.

        Returns:
            Array of unique reference directions.
        """
        ref_dirs: list[NDArray] = []
        for factory in self.layers:
            ref_dirs.append(factory)
        ref_dirs_arr = np.concatenate(ref_dirs, axis=0)
        is_duplicate = find_duplicates(ref_dirs_arr)
        return ref_dirs_arr[np.logical_not(is_duplicate)]


@default_random_state
def get_rng(random_state=None, **kwargs) -> object:  # type: ignore[misc]
    """Get random number generator.

    Args:
        random_state: Random state object.
        **kwargs: Additional arguments.

    Returns:
        The random state object.
    """
    return random_state


@default_random_state
def sample_on_unit_simplex(  # type: ignore[misc]
    n_points: int,
    n_dim: int,
    unit_simplex_mapping: str = "kraemer",
    random_state=None,
    **kwargs,
) -> NDArray:
    """Sample points on the unit simplex.

    Args:
        n_points: Number of points to sample.
        n_dim: Number of dimensions.
        unit_simplex_mapping: Mapping method ("sum", "kraemer", or "das-dennis").
        random_state: Random state for reproducibility.
        **kwargs: Additional arguments.

    Returns:
        Array of sampled points on unit simplex.

    Raises:
        Exception: If unit_simplex_mapping is invalid.
    """
    if unit_simplex_mapping == "sum":
        rnd = map_onto_unit_simplex(random_state.random((n_points, n_dim)), "sum")

    elif unit_simplex_mapping == "kraemer":
        rnd = map_onto_unit_simplex(random_state.random((n_points, n_dim)), "kraemer")

    elif unit_simplex_mapping == "das-dennis":
        n_partitions = get_partition_closest_to_points(n_points, n_dim)
        rnd = UniformReferenceDirectionFactory(n_dim, n_partitions=n_partitions).do()

    else:
        raise Exception("Please define a valid sampling on unit simplex strategy!")

    return rnd


def map_onto_unit_simplex(rnd: NDArray, method: str) -> NDArray:
    """Map points onto the unit simplex.

    Args:
        rnd: Array of points to map.
        method: Mapping method ("sum" or "kraemer").

    Returns:
        Array of points mapped onto unit simplex.

    Raises:
        Exception: If method is invalid.
    """
    n_points, n_dim = rnd.shape

    if method == "sum":
        ret = rnd / rnd.sum(axis=1)[:, None]

    elif method == "kraemer":
        M = sys.maxsize

        rnd *= M
        rnd = rnd[:, : n_dim - 1]
        rnd = np.column_stack([np.zeros(n_points), rnd, np.full(n_points, M)])

        rnd = np.sort(rnd, axis=1)

        ret = np.full((n_points, n_dim), np.nan)
        for i in range(1, n_dim + 1):
            ret[:, i - 1] = rnd[:, i] - rnd[:, i - 1]
        ret /= M

    else:
        raise Exception("Invalid unit simplex mapping!")

    return ret


def scale_reference_directions(ref_dirs: NDArray, scaling: float) -> NDArray:
    """Scale reference directions.

    Args:
        ref_dirs: Array of reference directions.
        scaling: Scaling factor.

    Returns:
        Scaled reference directions.
    """
    return ref_dirs * scaling + ((1 - scaling) / ref_dirs.shape[1])


def select_points_with_maximum_distance(
    X: NDArray,
    n_select: int,
    selected: list[int] | None = None,
    random_state=None,  # type: ignore[misc]
) -> list[int]:
    """Select points with maximum distance from each other.

    Args:
        X: Array of points.
        n_select: Number of points to select.
        selected: Initial selection (default: random).
        random_state: Random state for reproducibility.

    Returns:
        List of indices of selected points.
    """
    if selected is None:
        selected = []

    n_points, n_dim = X.shape

    # calculate the distance matrix
    D: NDArray = cdist(X, X)  # type: ignore[assignment]

    # if no selection provided pick randomly in the beginning
    if len(selected) == 0:
        # random_state should be provided by caller
        selected = [random_state.integers(len(X))]

    # create variables to store what selected and what not
    not_selected: list[int] = [i for i in range(n_points) if i not in selected]

    # remove unnecessary points
    dist_to_closest_selected = D[:, selected].min(axis=1)

    # now select the points until sufficient ones are found
    while len(selected) < n_select:
        # find point that has the maximum distance to all others
        index_in_not_selected = dist_to_closest_selected[not_selected].argmax()
        I = not_selected[index_in_not_selected]  # noqa: E741

        # add the closest distance to selected point
        is_closer = D[I] < dist_to_closest_selected
        dist_to_closest_selected[is_closer] = D[I][is_closer]

        # add it to the selected and remove from not selected
        selected.append(int(I))
        not_selected = [i for i, idx in enumerate(not_selected) if idx != index_in_not_selected]

    return selected
