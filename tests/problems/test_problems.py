import os

import pytest

from tests.util import run_usage, files_from_folder, USAGE, NO_INIT

SKIP = ["__init__.py", "usage_mopta08.py"]


@pytest.mark.parametrize('usage', files_from_folder(os.path.join(USAGE, "problems"), skip=SKIP))
def test_problems(usage):
    run_usage(usage)
    assert True
