import weakref

import numpy as np



class Tree:
    '''
    Implementation of Nary-tree.
    The source code is modified based on https://github.com/lianemeth/forest/blob/master/forest/NaryTree.py

    Parameters
    ----------
    key: object
        key of the node
    num_branch: int
        how many branches in each node
    children: Iterable[Tree]
        reference of the children
    parent: Tree
        reference of the parent node
    Returns
    -------
        an N-ary tree.
    '''

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
    """
    Tree-based efficient non-dominated sorting (T-ENS).
    This algorithm is very efficient in many-objective optimization problems (MaOPs).
    Parameters
    ----------
    F: np.array
        objective values for each individual.
    Returns
    -------
        indices of the individuals in each front.
    References
    ----------
    X. Zhang, Y. Tian, R. Cheng, and Y. Jin,
    A decision variable clustering based evolutionary algorithm for large-scale many-objective optimization,
    IEEE Transactions on Evolutionary Computation, 2018, 22(1): 97-112.
    """

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
    _, M = F.shape
    if forest[k] is None:
        forest[k] = Tree(key=p, num_branch=M - 1)
        left[p] = False
    elif check_tree(F, p, forest[k], obj_seq, True):
        left[p] = False


def check_tree(F, p, tree, obj_seq, add_pos):
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
