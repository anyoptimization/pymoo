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


@pytest.mark.xfail(raises=Exception)
def test_incremental_not_achievable_points():
    get_reference_directions("incremental", 3, n_points=110)


def test_unit_simplex_sampling():
    n_points = 1000
    n_dim = 3
    assert len(sample_on_unit_simplex(n_points, n_dim, unit_simplex_mapping="das-dennis")), 990
    assert len(sample_on_unit_simplex(n_points, n_dim, unit_simplex_mapping="sum")), 1000
    assert len(sample_on_unit_simplex(n_points, n_dim, unit_simplex_mapping="kraemer")), 1000
