import numpy as np
import pytest
import moocore

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting, find_non_dominated


CASES = {
    'empty':      np.zeros((0, 2)),
    'single':     np.array([[1.0, 2.0]]),
    'duplicates': np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
    'all_nd':     np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]),
    'all_dom':    np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    'mixed':      np.array([[1.0, 3.0], [2.0, 2.0], [2.0, 2.0], [3.0, 1.0], [4.0, 4.0]]),
    'random_2d':  np.random.RandomState(1).rand(50, 2),
    'random_3d':  np.random.RandomState(1).rand(50, 3),
    'random_5d':  np.random.RandomState(1).rand(50, 5),
}


@pytest.mark.parametrize('F', CASES.values(), ids=CASES.keys())
def test_find_non_dominated(F):
    pymoo = find_non_dominated(F)
    mc = np.array([], dtype=int) if len(F) == 0 else np.where(moocore.is_nondominated(F, keep_weakly=True))[0]
    np.testing.assert_array_equal(np.sort(pymoo), np.sort(mc))


@pytest.mark.parametrize('F', CASES.values(), ids=CASES.keys())
def test_non_dominated_sorting_ranks(F):
    if len(F) == 0:
        assert NonDominatedSorting().do(F) == []
        return
    _, pymoo_ranks = NonDominatedSorting().do(F, return_rank=True)
    mc_ranks = moocore.pareto_rank(F)
    np.testing.assert_array_equal(pymoo_ranks, mc_ranks)


def test_nds_n_stop_if_ranked():
    F = np.random.RandomState(1).rand(100, 3)
    fronts = NonDominatedSorting().do(F, n_stop_if_ranked=20)
    assert sum(len(f) for f in fronts) >= 20


def test_nds_only_first_front():
    F = np.random.RandomState(1).rand(50, 3)
    front0 = NonDominatedSorting().do(F, only_non_dominated_front=True)
    all_fronts = NonDominatedSorting().do(F)
    np.testing.assert_array_equal(np.sort(front0), np.sort(all_fronts[0]))
