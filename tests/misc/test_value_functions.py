import pytest

from pymoo.util import value_functions



@pytest.mark.parametrize('a,b,expect_sum', [(1, 2, 3), (2, 3, 5)])
def test_sum(a, b, expect_sum):
    assert value_functions.test_sum(a, b) == expect_sum




