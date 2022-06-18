import pytest

from tests.test_util import run_file, files_from_folder, EXAMPLES

SKIP = ["__init__.py",
        "video.py",
        "matlab_example.py",
        "stream.py",
        "mopta08.py",

        # SLOW
        # "usage_gif.py",
        # "usage_nds.py",
        # "usage_energy.py",
        # "usage_energy.py",

        "__pycache__"]


@pytest.mark.parametrize('example', files_from_folder(EXAMPLES, skip=SKIP))
def test_examples(example):
    run_file(example)
    assert True
