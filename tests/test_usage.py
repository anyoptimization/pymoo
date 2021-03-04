import pytest

from tests.util import run_usage, files_from_folder, USAGE

SKIP = ["__init__.py", "usage_video.py", "usage_matlab.py", "usage_stream.py", "usage_mopta08.py", "__pycache__"]


@pytest.mark.parametrize('usage', files_from_folder(USAGE, skip=SKIP))
def test_usage(usage):
    run_usage(usage)
    assert True
