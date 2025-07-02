import numpy as np

from pymoo.util.misc import swap
from pymoo.util import default_random_state


@default_random_state
def randomized_argsort(A, method="numpy", order='ascending', random_state=None):
    if method == "numpy":
        P = random_state.permutation(len(A))
        I = np.argsort(A[P], kind='quicksort')
        I = P[I]

    elif method == "quicksort":
        I = quicksort(A)

    else:
        raise Exception("Randomized sort method not known.")

    if order == 'ascending':
        return I
    elif order == 'descending':
        return np.flip(I, axis=0)
    else:
        raise Exception("Unknown sorting order: ascending or descending.")


@default_random_state
def quicksort(A, random_state=None):
    I = np.arange(len(A))
    _quicksort(A, I, 0, len(A) - 1, random_state=random_state)
    return I


def _quicksort(A, I, left, right, random_state):
    if left < right:

        index = random_state.integers(left, right + 1)
        swap(I, right, index)

        pivot = A[I[right]]

        i = left - 1

        for j in range(left, right):

            if A[I[j]] <= pivot:
                i += 1
                swap(I, i, j)

        index = i + 1
        swap(I, right, index)

        _quicksort(A, I, left, index - 1, random_state)
        _quicksort(A, I, index + 1, right, random_state)


if __name__ == '__main__':
    a = np.array([5, 9, 10, 0, 0, 0, 100, -2])

    for i in range(200):
        I = randomized_argsort(a, method="numpy")
        print(I)
