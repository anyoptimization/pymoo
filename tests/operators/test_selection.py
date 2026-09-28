import numpy as np
import pytest

from pymoo.operators.selection.rnd import fast_fill_random


@pytest.mark.parametrize("seed", range(5))
def test_fast_fill_random_distinct_and_not_itself(seed):
    # as DE/rand/1 uses it: three indices per target, distinct and different from the target
    n = 100
    itself = np.arange(n)[:, None]
    random_state = np.random.default_rng(seed)

    for _ in range(50):
        P = np.full((n, 3), -1)
        fast_fill_random(P, n, columns=range(3), Xp=itself, random_state=random_state)

        rows = np.column_stack([P, itself])
        assert np.all((rows >= 0) & (rows < n))
        assert all(len(set(row)) == rows.shape[1] for row in rows)
