import pytest

from tests.test_util import run_file, files_from_folder, EXAMPLES

SKIP = ["__init__.py",
        "video.py",
        "matlab_example.py",
        "stream.py",
        "mopta08.py",
        "coco.py",
        "gif.py",
        "__pycache__"]


@pytest.mark.examples
@pytest.mark.long
@pytest.mark.parametrize('example', files_from_folder(EXAMPLES, skip=SKIP))
def test_examples(example):
    run_file(example)
    assert True
