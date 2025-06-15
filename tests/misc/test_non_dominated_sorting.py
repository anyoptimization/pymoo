import numpy as np
import pytest

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.problems.many import DTLZ2
from pymoo.functions import load_function
from pymoo.util.ref_dirs import get_reference_directions


def assert_fronts_equal(fronts_a, fronts_b):
    assert len(fronts_a) == len(fronts_b)

    for a, b in zip(fronts_a, fronts_b):
        assert len(a) == len(b)
        assert set(a) == set(b)


def test_fast_non_dominated_sorting():
    F = np.random.random((100, 2))
    fronts = load_function("fast_non_dominated_sort", _type="python")(F)
    _fronts = load_function("fast_non_dominated_sort", _type="cython")(F)
    assert_fronts_equal(fronts, _fronts)


def test_efficient_non_dominated_sort():
    print("Testing ENS...")
    F = np.ones((1000, 3))
    F[:, 1:] = np.random.random((1000, 2))

    nds = load_function("fast_non_dominated_sort", _type="python")(F)

    python_fronts_seq = load_function("efficient_non_dominated_sort", _type="python")(F)
    cython_fronts_seq = load_function("efficient_non_dominated_sort", _type="cython")(F)

    assert_fronts_equal(nds, python_fronts_seq)
    assert_fronts_equal(nds, cython_fronts_seq)

    python_fronts_binary = load_function(
        "efficient_non_dominated_sort", _type="python"
    )(F, strategy="binary")
    cython_fronts_binary = load_function(
        "efficient_non_dominated_sort", _type="cython"
    )(F, strategy="binary")

    assert_fronts_equal(nds, python_fronts_binary)
    assert_fronts_equal(nds, cython_fronts_binary)


def test_tree_based_non_dominated_sort():
    print("Testing T-ENS...")
    F = np.ones((1000, 3))
    F[:, 1:] = np.random.random((1000, 2))
    _fronts = load_function("fast_non_dominated_sort", _type="python")(F)

    fronts = load_function("tree_based_non_dominated_sort", _type="python")(F)
    assert_fronts_equal(_fronts, fronts)


def test_dominance_degree_non_dominated_sort():
    print("Testing DDA-NS/DDA-ENS...")
    F = np.ones((1000, 3))
    F[:, 1:] = np.random.random((1000, 2))

    nds = load_function("fast_non_dominated_sort", _type="python")(F)

    python_fronts_fast = load_function(
        "dominance_degree_non_dominated_sort", _type="python"
    )(F, strategy="fast")
    cython_fronts_fast = load_function(
        "dominance_degree_non_dominated_sort", _type="cython"
    )(F, strategy="fast")

    assert_fronts_equal(nds, python_fronts_fast)
    assert_fronts_equal(nds, cython_fronts_fast)

    python_fronts_seq = load_function(
        "dominance_degree_non_dominated_sort", _type="python"
    )(F, strategy="efficient")
    cython_fronts_seq = load_function(
        "dominance_degree_non_dominated_sort", _type="cython"
    )(F, strategy="efficient")

    assert_fronts_equal(nds, python_fronts_seq)
    assert_fronts_equal(nds, cython_fronts_seq)


class MyCallback(Callback):
    def notify(self, algorithm):
        F = algorithm.pop.get("F")

        python_fast_nds = load_function("fast_non_dominated_sort", _type="python")(F)
        cython_fast_nds = load_function("fast_non_dominated_sort", _type="cython")(F)
        assert_fronts_equal(python_fast_nds, cython_fast_nds)

        python_efficient_fast_nds = load_function(
            "efficient_non_dominated_sort", _type="python"
        )(F, strategy="binary")
        assert_fronts_equal(python_fast_nds, python_efficient_fast_nds)

        cython_efficient_fast_nds = load_function(
            "efficient_non_dominated_sort", _type="cython"
        )(F, strategy="binary")
        assert_fronts_equal(python_efficient_fast_nds, cython_efficient_fast_nds)

        python_efficient_fast_nds_bin = load_function(
            "efficient_non_dominated_sort", _type="python"
        )(F)
        assert_fronts_equal(python_fast_nds, python_efficient_fast_nds_bin)

        cython_efficient_fast_nds_bin = load_function(
            "efficient_non_dominated_sort", _type="cython"
        )(F)
        assert_fronts_equal(
            python_efficient_fast_nds_bin, cython_efficient_fast_nds_bin
        )

        python_tree_based_nds = load_function(
            "tree_based_non_dominated_sort", _type="python"
        )(F)
        assert_fronts_equal(python_fast_nds, python_tree_based_nds)

        python_dda_fast_nds_ens = load_function(
            "dominance_degree_non_dominated_sort", _type="python"
        )(F, strategy="efficient")
        assert_fronts_equal(python_fast_nds, python_dda_fast_nds_ens)

        cython_dda_fast_nds_ens = load_function(
            "dominance_degree_non_dominated_sort", _type="cython"
        )(F, strategy="efficient")
        assert_fronts_equal(python_fast_nds, cython_dda_fast_nds_ens)

        python_dda_fast_nds_ns = load_function(
            "dominance_degree_non_dominated_sort", _type="python"
        )(F, strategy="fast")
        assert_fronts_equal(python_fast_nds, python_dda_fast_nds_ns)

        cython_dda_fast_nds_ns = load_function(
            "dominance_degree_non_dominated_sort", _type="cython"
        )(F, strategy="fast")
        assert_fronts_equal(python_fast_nds, cython_dda_fast_nds_ns)


@pytest.mark.long
@pytest.mark.parametrize("n_obj", [2, 3, 5, 10])
def test_equal_during_run(n_obj):
    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("energy", n_obj, n_points=100)

    # create the algorithm object
    algorithm = NSGA3(pop_size=92, ref_dirs=ref_dirs)

    print(f"NDS with {n_obj} objectives.")

    # execute the optimization
    minimize(
        DTLZ2(n_obj=n_obj),
        algorithm,
        callback=MyCallback(),
        seed=1,
        termination=("n_gen", 200),
        verbose=True,
    )
