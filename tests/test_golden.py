"""Golden behavior-regression snapshots across the algorithm portfolio.

The complement to the deterministic suites: those prove an algorithm reproduces
*itself* run-to-run (same seed → same result); these prove its output never moves
*across refactors* by snapshotting it to a committed baseline.

A `@pytest.mark.golden` test *returns* its observable output (`res.F`); pyclawd's
golden plugin (a dev dependency) captures it and compares against the committed
baseline under `tests/golden/`, with the tolerances from `GoldenConfig`.

The algorithm lists + instantiation helper are reused from the deterministic
suites so this golden portfolio stays in sync with them automatically.

    pyclawd golden                 # compare against committed baselines
    pyclawd golden update          # re-record (humans bless), then git diff + commit
    pyclawd golden status          # inventory; flag orphaned baselines

`golden` is its own tier — excluded from the unit suite (`pyclawd test`/`check`)
and run as a separate behavior gate.
"""

import pytest

from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT1
from pymoo.problems.single import Sphere
from tests.algorithms.test_deterministic_moo import (
    MULTI_OBJECTIVE_ALGORITHM_CLASSES,
    create_algorithm_instance,
)
from tests.algorithms.test_deterministic_soo import ALL_SINGLE_OBJECTIVE_ALGORITHMS

SEED = 42
N_GEN = 10
SOO_PROBLEM = Sphere(n_var=2)
MOO_PROBLEM = ZDT1()


@pytest.mark.golden
@pytest.mark.parametrize(
    "algorithm_class",
    ALL_SINGLE_OBJECTIVE_ALGORITHMS,
    ids=[a.__name__ for a in ALL_SINGLE_OBJECTIVE_ALGORITHMS],
)
def test_golden_soo(algorithm_class):
    """Single-objective algorithm output on Sphere must not drift across refactors."""
    res = minimize(
        SOO_PROBLEM, algorithm_class(), ("n_gen", N_GEN), seed=SEED, verbose=False
    )
    return res.F


@pytest.mark.golden
@pytest.mark.parametrize(
    "algorithm_class",
    MULTI_OBJECTIVE_ALGORITHM_CLASSES,
    ids=[a.__name__ for a in MULTI_OBJECTIVE_ALGORITHM_CLASSES],
)
def test_golden_moo(algorithm_class):
    """Multi-objective algorithm front on ZDT1 must not drift across refactors."""
    algorithm = create_algorithm_instance(algorithm_class)
    res = minimize(MOO_PROBLEM, algorithm, ("n_gen", N_GEN), seed=SEED, verbose=False)
    return res.F
