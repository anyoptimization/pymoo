import os
from os.path import join, dirname, exists

import numpy as np
import pytest

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.core.callback import Callback
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions


class ModificationTestCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data = []

    def _update(self, algorithm):
        super()._update(algorithm)

        igd = IGD(algorithm.problem.pareto_front())
        F = algorithm.pop.get("F")
        self.data.append(igd.do(F))


ref_dirs = get_reference_directions("uniform", 2, n_partitions=99)

RUNS = [
    ("nsga2", NSGA2()),
    ("nsga3", NSGA3(ref_dirs)),
    ("rvea", RVEA(ref_dirs)),
]

PROBLEMS = ["zdt1", "zdt2", "zdt3"]

HASH = join(dirname(__file__), "hash")

@pytest.mark.skip(reason="this has changed. unclear why at this point.")
@pytest.mark.parametrize('entry', RUNS)
@pytest.mark.parametrize('p', PROBLEMS)
def test_no_mod(entry, p):
    name, algorithm = entry

    problem = get_problem(p)

    callback = ModificationTestCallback()

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 200),
                   callback=callback,
                   seed=1,
                   verbose=False)

    igd = np.array(callback.data)

    FOLDER = join(HASH, name)
    if not exists(FOLDER):
        os.makedirs(FOLDER)

    FNAME = join(FOLDER, p)

    # BE VERY CAREFUL WITH THIS LINE. THIS OVERRIDES ALL EXISTING HASHES
    # np.save(FNAME, igd)

    # correct = int(open(FNAME, 'r').read())
    correct = np.load(FNAME + ".npy")

    np.testing.assert_almost_equal(igd, correct)
