import pickle

import numpy as np
import pytest

from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from tests.test_util import load_to_test_resource


@pytest.fixture
def data():
    return pickle.load(open(load_to_test_resource('cnsga2_zdt4.dat'), 'rb'))


def test_normalization():
    F = np.array([[0, 0], [0.5, 50], [1.0, 100]])
    cd = calc_crowding_distance(F)
    np.testing.assert_almost_equal(cd, np.array([np.inf, 1.0, np.inf]))


def test_rank_and_crowding_distance(data):
    for i, D in enumerate(data):

        survivor_and_last_front = np.where(D['rank'] != -1.0)[0]
        crowding = D['crowding'][survivor_and_last_front]
        rank = D['rank'][survivor_and_last_front].astype(int)
        F = D['F'][survivor_and_last_front, :]

        fronts, _rank = NonDominatedSorting().do(F, return_rank=True)
        _rank += 1
        _crowding = np.full(len(F), np.nan)
        for front in fronts:
            _crowding[front] = calc_crowding_distance(F[front])
        _crowding[np.isinf(_crowding)] = 1e14

        is_equal = np.all(rank == _rank)
        if not is_equal:
            index = np.where(rank == _rank)
            print(index)
            print(D['rank'][index])
            print(D['F'][index])

        assert is_equal

        is_equal = np.all(np.abs(_crowding - crowding) < 0.001)
        if not is_equal:

            index = np.where(np.abs(_crowding - crowding) > 0.001)[0]
            index = index[np.argsort(rank[index])]

            # only an error if it is not a duplicate F value
            for i_not_equal in index:

                if len(np.where(np.all(F[i_not_equal, :] == F, axis=1))[0]) == 1:
                    print("-" * 30)
                    print("Generation: ", i)
                    print("Is rank equal: ", np.all(rank == _rank))

                    print(index)
                    print(rank[index])
                    print(F[index])
                    print(np.concatenate([_crowding[:, None], crowding[:, None]], axis=1)[index, :])
                    print()

                    assert is_equal
