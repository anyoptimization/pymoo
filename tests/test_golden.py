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

import json
import platform
from pathlib import Path

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

BASELINE_PATH = Path(__file__).parent / "golden" / "test_golden.json"

# These baselines snapshot the *exact* ``res.F`` of a full optimization run. That
# value is only bit-reproducible on the platform it was recorded on: population
# EAs amplify the last-bit differences between platform math libraries (e.g. the
# transcendental ``pow`` in SBX/PM crossover-mutation differs between glibc and
# Apple's libm) into a completely different — but equally valid — Pareto front
# within a single generation. No tolerance bridges that gap, so the golden tier
# is pinned to the platform the baselines live on (linux-x86_64, which CI runs)
# and *skips* elsewhere rather than reporting spurious failures. Contributors on
# other platforms get the golden gate via CI; see GoldenConfig in .pyclawd/config.py.
BASELINE_PLATFORM = ("Linux", "x86_64")
pytestmark = pytest.mark.skipif(
    (platform.system(), platform.machine()) != BASELINE_PLATFORM,
    reason=f"golden res.F baselines are recorded on {BASELINE_PLATFORM}; "
    "res.F is platform-chaotic for population EAs — run this tier via CI",
)


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


def test_golden_coverage():
    """Every parametrized algorithm must own a committed golden baseline.

    A regular (non-``golden``) unit test so it gates every PR: it guards the gate
    itself. Adding an algorithm to ``ALL_SINGLE_OBJECTIVE_ALGORITHMS`` /
    ``MULTI_OBJECTIVE_ALGORITHM_CLASSES`` without blessing a baseline (``pyclawd
    golden update``) fails here — so a new algorithm can never silently ship with no
    behavior-regression net. Also flags orphaned baselines (a removed/renamed
    algorithm leaving a stale key behind).
    """
    assert BASELINE_PATH.exists(), (
        f"golden baseline file is missing ({BASELINE_PATH}); record it with "
        f"`pyclawd golden update`"
    )
    recorded = set(json.loads(BASELINE_PATH.read_text()).keys())

    expected = {f"test_golden_soo[{a.__name__}]" for a in ALL_SINGLE_OBJECTIVE_ALGORITHMS}
    expected |= {f"test_golden_moo[{a.__name__}]" for a in MULTI_OBJECTIVE_ALGORITHM_CLASSES}

    missing = sorted(expected - recorded)
    orphaned = sorted(recorded - expected)
    assert not missing, (
        f"algorithms without a committed golden baseline: {missing}. "
        f"Bless them with `pyclawd golden update`."
    )
    assert not orphaned, (
        f"stale golden baselines with no matching algorithm: {orphaned}. "
        f"Remove them (e.g. re-record with `pyclawd golden update`)."
    )
