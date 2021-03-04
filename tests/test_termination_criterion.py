import os

import pytest

from tests.util import run_usage, files_from_folder, USAGE, NO_INIT


@pytest.mark.parametrize('usage', files_from_folder(os.path.join(USAGE, "termination"), skip=NO_INIT))
def test_termination_criterion(usage):
    run_usage(usage)
    assert True
