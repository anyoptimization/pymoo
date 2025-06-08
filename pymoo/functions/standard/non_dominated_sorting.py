"""
Standard Python implementations of non-dominated sorting algorithms.
"""

import numpy as np
from math import floor
import weakref
from typing import Literal, List

from pymoo.util.dominator import Dominator


def fast_non_dominated_sort(F, dominator=Dominator(), **kwargs):
    """Fast non-dominated sorting algorithm."""
    if "dominator" in kwargs:
        M = Dominator.calc_domination_matrix(F)
    else:
        M = dominator.calc_domination_matrix(F)

    # calculate the dominance matrix
    n = M.shape[0]

    fronts = []

    if n == 0:
        return fronts

    # final rank that will be returned
    n_ranked = 0
    ranked = np.zeros(n, dtype=int)

    # for each individual a list of all individuals that are dominated by this one
    is_dominating = [[] for _ in range(n)]

    # storage for the number of solutions dominated this one
    n_dominated = np.zeros(n)

    current_front = []

    for i in range(n):

        for j in range(i + 1, n):
            rel = M[i, j]
            if rel == 1:
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif rel == -1:
                is_dominating[j].append(i)
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            current_front.append(i)
            ranked[i] = 1.0
            n_ranked += 1

    # append the first front to the current front
    fronts.append(current_front)

    # while not all solutions are assigned to a pareto front
    while n_ranked < n:

        next_front = []

        # for each individual in the current front
        for i in current_front:

            # all solutions that are dominated by this individuals
            for j in is_dominating[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1.0
                    n_ranked += 1

        fronts.append(next_front)
        current_front = next_front

    return fronts


def find_non_dominated(F, epsilon=0.0):
    """
    Simple and efficient implementation to find only non-dominated points.
    Uses straightforward O(n²) algorithm with early termination.
    """
    n_points = F.shape[0]
    non_dominated_indices = []
    
    if n_points == 0:
        return np.array([], dtype=int)
    
    # Check each point to see if it's non-dominated
    for i in range(n_points):
        is_dominated = False
        
        # Check if point i is dominated by any other point j
        for j in range(n_points):
            if i != j:
                # Check if j dominates i
                dominates = True
                at_least_one_better = False
                
                for k in range(F.shape[1]):  # for each objective
                    if F[j, k] + epsilon < F[i, k]:  # j is better than i in objective k
                        at_least_one_better = True
                    elif F[j, k] > F[i, k] + epsilon:  # j is worse than i in objective k
                        dominates = False
                        break  # Early termination in objective loop
                
                # j dominates i if j is at least as good in all objectives and better in at least one
                if dominates and at_least_one_better:
                    is_dominated = True
                    break  # Early termination - no need to check other points
        
        # If point i is not dominated by any other point, it's non-dominated
        if not is_dominated:
            non_dominated_indices.append(i)
    
    return np.array(non_dominated_indices, dtype=int)


def efficient_non_dominated_sort(F, strategy="sequential"):
    """Efficient Non-dominated Sorting (ENS)"""
    assert (strategy in ["sequential", 'binary']), "Invalid search strategy"

    # the shape of the input
    N, M = F.shape

    # do a lexicographic ordering
    I = np.lexsort(F.T[::-1])
    F = F[I]

    # front ranks for each individual
    fronts = []

    for i in range(N):

        if strategy == 'sequential':
            k = sequential_search(F, i, fronts)
        else:
            k = binary_search(F, i, fronts)

        # create empty fronts if necessary
        if k >= len(fronts):
            fronts.append([])

        # append the current individual to a front
        fronts[k].append(i)

    # now map the fronts back to the originally sorting
    ret = []
    for front in fronts:
        ret.append(I[front])

    return ret


def sequential_search(F, i, fronts) -> int:
    """Find the front rank for the i-th individual through sequential search."""
    num_found_fronts = len(fronts)
    k = 0  # the front now checked
    current = F[i]
    while True:
        if num_found_fronts == 0:
            return 0
        # solutions in the k-th front, examine in reverse order
        fk_indices = fronts[k]
        solutions = F[fk_indices[::-1]]
        non_dominated = True
        for f in solutions:
            relation = Dominator.get_relation(current, f)
            if relation == -1:
                non_dominated = False
                break
        if non_dominated:
            return k
        else:
            k += 1
            if k >= num_found_fronts:
                # move the individual to a new front
                return num_found_fronts


def binary_search(F, i, fronts):
    """Find the front rank for the i-th individual through binary search."""
    num_found_fronts = len(fronts)
    if num_found_fronts == 0:
        return 0

    k_min = 0  # the lower bound for checking
    k_max = num_found_fronts  # the upper bound for checking
    k = floor((k_max + k_min) / 2 + 0.5)  # the front now checked
    current = F[i]
    while True:

        # solutions in the k-th front, examine in reverse order
        fk_indices = fronts[k - 1]
        solutions = F[fk_indices[::-1]]
        non_dominated = True

        for f in solutions:
            relation = Dominator.get_relation(current, f)
            if relation == -1:
                non_dominated = False
                break

        # binary search
        if non_dominated:
            if k == k_min + 1:
                return k - 1
            else:
                k_max = k
                k = floor((k_max + k_min) / 2 + 0.5)
        else:
            k_min = k
            if k_max == k_min + 1 and k_max < num_found_fronts:
                return k_max - 1
            elif k_min == num_found_fronts:
                return num_found_fronts
            else:
                k = floor((k_max + k_min) / 2 + 0.5)


class Tree:
    """Implementation of N-ary tree for tree-based non-dominated sorting."""
    
    def __init__(self, key, num_branch, children=None, parent=None):
        self.key = key
        self.children = children or [None for _ in range(num_branch)]
        self._parent = weakref.ref(parent) if parent else None

    @property
    def parent(self):
        if self._parent:
            return self._parent()

    def __getstate__(self):
        self._parent = None

    def __setstate__(self, state):
        self.__dict__ = state
        for child in self.children:
            child._parent = weakref.ref(self)

    def traversal(self, visit=None, *args, **kwargs):
        if visit is not None:
            visit(self, *args, **kwargs)
        l = [self]
        for child in self.children:
            if child is not None:
                l += child.traversal(visit, *args, **kwargs)
        return l


def tree_based_non_dominated_sort(F):
    """Tree-based efficient non-dominated sorting (T-ENS)."""
    N, M = F.shape
    # sort the rows in F
    indices = np.lexsort(F.T[::-1])
    F = F[indices]

    obj_seq = np.argsort(F[:, :0:-1], axis=1) + 1

    k = 0

    forest = []

    left = np.full(N, True)
    while np.any(left):
        forest.append(None)
        for p, flag in enumerate(left):
            if flag:
                update_tree(F, p, forest, k, left, obj_seq)
        k += 1

    # convert forest to fronts
    fronts = [[] for _ in range(k)]
    for k, tree in enumerate(forest):
        fronts[k].extend([indices[node.key] for node in tree.traversal()])
    return fronts


def update_tree(F, p, forest, k, left, obj_seq):
    """Update tree for tree-based non-dominated sorting."""
    _, M = F.shape
    if forest[k] is None:
        forest[k] = Tree(key=p, num_branch=M - 1)
        left[p] = False
    elif check_tree(F, p, forest[k], obj_seq, True):
        left[p] = False


def check_tree(F, p, tree, obj_seq, add_pos):
    """Check tree for tree-based non-dominated sorting."""
    if tree is None:
        return True

    N, M = F.shape

    # find the minimal index m satisfying that p[obj_seq[tree.root][m]] < tree.root[obj_seq[tree.root][m]]
    m = 0
    while m < M - 1 and F[p, obj_seq[tree.key, m]] >= F[tree.key, obj_seq[tree.key, m]]:
        m += 1

    # if m not found
    if m == M - 1:
        # p is dominated by the solution at the root
        return False
    else:
        for i in range(m + 1):
            # p is dominated by a solution in the branch of the tree
            if not check_tree(F, p, tree.children[i], obj_seq, i == m and add_pos):
                return False

        if tree.children[m] is None and add_pos:
            # add p to the branch of the tree
            tree.children[m] = Tree(key=p, num_branch=M - 1)
        return True


def construct_comp_matrix(vec: np.ndarray, sorted_idx: np.ndarray) -> np.ndarray:
    """Construct the comparison matrix from a row-vector vec."""
    n = vec.shape[0]
    c = np.zeros(shape=(n, n), dtype=np.int32)

    # the elements of the b(0)-th row in C are all set to 1
    c[sorted_idx[0], :] = 1

    for i in range(1, n):
        if vec[sorted_idx[i]] == vec[sorted_idx[i - 1]]:
            # the rows in C corresponding to the same elements in w are identical
            c[sorted_idx[i]] = c[sorted_idx[i - 1]]
        else:
            c[sorted_idx[i], sorted_idx[i:]] = 1

    return c


def construct_domination_matrix(f_scores: np.ndarray, **kwargs) -> np.ndarray:
    """Calculate the dominance degree matrix for a set of vectors."""
    d = np.zeros((f_scores.shape[0], f_scores.shape[0]), dtype=np.int32)
    b = np.apply_over_axes(np.argsort, f_scores, axes=0)
    for vec, srt in zip(f_scores.T, b.T):
        d += construct_comp_matrix(vec, srt)
    d = np.where(
        np.logical_and(d == f_scores.shape[-1], d.T == f_scores.shape[-1]), 0, d
    )
    return d


def dda_ns(f_scores: np.ndarray, **kwargs) -> List[List[int]]:
    """DDA-NS algorithm."""
    d_mx = construct_domination_matrix(f_scores)
    max_d = np.empty((f_scores.shape[0],), dtype=np.int32)

    fronts = []
    count = 0
    while count < f_scores.shape[0]:
        # Max(D) is the row vector containing the maximum elements from each column of D
        np.max(d_mx, out=max_d, axis=0)
        front = [i for i, m_d in enumerate(max_d) if 0 <= m_d < f_scores.shape[-1]]
        count += len(front)
        d_mx[front] = -1
        d_mx[:, front] = -1
        fronts.append(front)

    return fronts


def dda_ens(f_scores: np.ndarray, **kwargs) -> List[List[int]]:
    """DDA-ENS (efficient DDA) algorithm."""
    d_mx = construct_domination_matrix(f_scores)

    fronts: List[List[int]] = []
    for s in np.lexsort(f_scores.T):
        isinserted = False
        for fk in fronts:
            if not (d_mx[fk, s] == f_scores.shape[1]).any():
                fk.append(s)
                isinserted = True
                break
        if not isinserted:
            fronts.append([s])
    return fronts


def dominance_degree_non_dominated_sort(
    f_scores: np.ndarray, strategy: Literal["efficient", "fast"] = "efficient"
) -> List[List[int]]:
    """Perform non-dominating sort with the specified algorithm."""
    if strategy == "efficient":
        return dda_ens(f_scores)
    if strategy == "fast":
        return dda_ns(f_scores)
    raise ValueError("Invalid search strategy")


def fast_best_order_sort(*args, **kwargs):
    """Placeholder for fast_best_order_sort - only available in Cython."""
    raise NotImplementedError("fast_best_order_sort is only available in compiled (Cython) version")