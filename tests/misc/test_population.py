from copy import deepcopy

from pymoo.core.individual import Individual
from pymoo.core.population import Population

import numpy as np


def test_init():
    pop = Population.empty(100)
    assert len(pop) == 100

    pop = Population([Individual() for _ in range(50)])
    assert len(pop) == 50


def test_copy():
    a = Population.new(X=np.random.random((100, 2)))
    b = deepcopy(a)
    b[0].X[:] = -1

    assert a[0] != b[0]
    assert np.all(a[0].X != b[0].X)


def test_has_method():
    pop = Population.empty(100)
    assert pop.has("X")
    assert not pop.has("opt")

    pop[:-1].set("opt", False)
    assert not pop.has("opt")

    pop[-1].set("opt", True)
    assert pop.has("opt")
