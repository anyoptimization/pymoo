"""Tests for the COMO-CMA-ES wrapper (requires the optional comocma package)."""

import numpy as np
import pytest

pytest.importorskip("comocma")

from pymoo.algorithms.moo.como_cmaes import COMOCMAES  # noqa: E402
from pymoo.indicators.hv import Hypervolume  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402
from pymoo.problems import get_problem  # noqa: E402


def test_como_cmaes_runs_and_is_competitive():
    problem = get_problem("zdt1", n_var=10)
    res = minimize(problem, COMOCMAES(pop_size=50, reference_point=[1.1, 1.1]),
                   ("n_evals", 20000), seed=1, verbose=False)
    assert len(res.F) > 1
    hv = Hypervolume(ref_point=np.array([1.1, 1.1]))(res.F)
    assert hv > 0.75  # close to a well-converged NSGA-II front


def test_como_cmaes_requires_biobjective():
    problem = get_problem("dtlz2", n_var=7, n_obj=3)
    with pytest.raises(ValueError, match="bi-objective"):
        minimize(problem, COMOCMAES(pop_size=10), ("n_gen", 2), seed=1, verbose=False)


def test_como_cmaes_rejects_constrained_problems():
    # comocma only sees F, so constraints would be silently ignored and the
    # returned solutions reported with CV = 0
    problem = get_problem("tnk")
    with pytest.raises(ValueError, match="constrained"):
        minimize(problem, COMOCMAES(pop_size=10), ("n_gen", 2), seed=1, verbose=False)


def test_como_cmaes_deterministic():
    problem = get_problem("zdt1", n_var=5)
    r1 = minimize(problem, COMOCMAES(pop_size=20, reference_point=[1.1, 1.1]),
                  ("n_gen", 20), seed=5, verbose=False)
    r2 = minimize(problem, COMOCMAES(pop_size=20, reference_point=[1.1, 1.1]),
                  ("n_gen", 20), seed=5, verbose=False)
    assert r1.F.shape == r2.F.shape
    np.testing.assert_allclose(r1.F, r2.F)
