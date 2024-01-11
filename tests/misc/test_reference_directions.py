import pytest

from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.reference_direction import sample_on_unit_simplex


def test_das_dennis():
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    assert len(ref_dirs) == 91


def test_energy():
    ref_dirs = get_reference_directions("energy", 3, n_points=100, verify_gradient=True)
    assert len(ref_dirs) == 100


def test_das_dennis_achievable_points():
    ref_dirs = get_reference_directions("das-dennis", 3, n_points=91)
    assert len(ref_dirs) == 91


@pytest.mark.xfail(raises=Exception)
def test_das_dennis_not_achievable_points():
    get_reference_directions("das-dennis", 3, n_points=92)


def test_incremental():
    ref_dirs = get_reference_directions("incremental", 3, n_partitions=8)
    assert len(ref_dirs) == 109

def test_incremental_achievable_points():
    ref_dirs = get_reference_directions("incremental", 3, n_points=109)
    assert len(ref_dirs) == 109

def test_incremental2():
    N = [[], [], 
         [1, 3,  5,   7,   9,   11,   13,   15,    17,    19],
         [1, 4, 10,  19,  31,   46,   64,   85,   109,   136],
         [1, 5, 15,  35,  69,  121,  195,  295,   425,   589],
         [1, 6, 21,  56, 126,  251,  456,  771,  1231,  1876],
         [1, 7, 28,  84, 210,  462,  923, 1709,  2975,  4921],
         [1, 8, 36, 120, 330,  792, 1716, 3431,  6427, 11404],
         [1, 9, 45, 165, 495, 1287, 3003, 6435, 12869, 24301]]
    for i, list in enumerate(N):
        for j, x in enumerate(list):
            ref_dirs = get_reference_directions("incremental", i, n_partitions=j)
            assert len(ref_dirs) == x
            ref_dirs = get_reference_directions("incremental", i, n_points=x)
            assert len(ref_dirs) == x


@pytest.mark.xfail(raises=Exception)
def test_incremental_not_achievable_points():
    get_reference_directions("incremental", 3, n_points=110)


def test_unit_simplex_sampling():
    n_points = 1000
    n_dim = 3
    assert len(sample_on_unit_simplex(n_points, n_dim, unit_simplex_mapping="das-dennis")), 990
    assert len(sample_on_unit_simplex(n_points, n_dim, unit_simplex_mapping="sum")), 1000
    assert len(sample_on_unit_simplex(n_points, n_dim, unit_simplex_mapping="kraemer")), 1000
