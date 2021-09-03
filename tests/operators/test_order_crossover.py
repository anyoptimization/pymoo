import numpy as np

from pymoo.operators.crossover.ox import ox


def order_crossover_contributed_no_shift(x1, x2, seq=None):
    assert len(x1) == len(x2)

    if seq is not None:
        start, end = seq
    else:
        start, end = np.sort(np.random.choice(len(x1), 2, replace=False))

    y1 = x1.copy()
    y2 = x2.copy()
    # build y1 and y2
    segment1 = set(y1[start:end])
    segment2 = set(y2[start:end])
    I = np.concatenate((np.arange(0, start), np.arange(end, len(x1))))

    # find elements in x2 that are not in segment1
    y1[I] = [y for y in x2 if y not in segment1]
    # find elements in x1 that are not in segment2
    y2[I] = [y for y in x1 if y not in segment2]

    return y1, y2


def test_example_from_goldberg():
    a = np.array([9, 8, 4, 5, 6, 7, 1, 3, 2, 0])
    b = np.array([8, 7, 1, 2, 3, 0, 9, 5, 4, 6])

    np.testing.assert_allclose(ox(a, b, (3, 5), shift=True), np.array([5, 6, 7, 2, 3, 0, 1, 9, 8, 4]))
    np.testing.assert_allclose(ox(b, a, (3, 5), shift=True), np.array([2, 3, 0, 5, 6, 7, 9, 4, 8, 1]))


# http://mat.uab.cat/~alseda/MasterOpt/GeneticOperations.pdf
def test_other_example():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = np.array([5, 7, 4, 9, 1, 3, 6, 2, 8])
    res = ox(b, a, (2, 5), shift=False)

    np.testing.assert_allclose(res, np.array([7, 9, 3, 4, 5, 6, 1, 2, 8]))


def test_example_to_bound():
    a = np.array([9, 8, 4, 5, 6, 7, 1, 3, 2, 0])
    b = np.array([8, 7, 1, 2, 3, 0, 9, 5, 4, 6])
    np.testing.assert_allclose(ox(a, b, (3, len(b)), shift=False), np.array([8, 7, 1, 2, 3, 0, 9, 5, 4, 6]))


def test_equal_constribution_no_shift():
    for _ in range(100):
        a = np.random.permutation(10)
        b = np.random.permutation(10)

        start, end = np.sort(np.random.choice(len(a), 2, replace=False))
        y1 = ox(a, b, seq=(start, end), shift=False)
        y2 = ox(b, a, seq=(start, end), shift=False)

        _y1, _y2 = order_crossover_contributed_no_shift(a, b, seq=(start, end + 1))

        np.testing.assert_allclose(_y1, y2)
        np.testing.assert_allclose(_y2, y1)
