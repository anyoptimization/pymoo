import os

import pytest

from pymoo.configuration import get_pymoo
from tests.util import files_from_folder, run_ipynb

DOCS = os.path.join(get_pymoo(), "doc", "source")

OVERWRITE = False

SKIP = ["parallelization.ipynb"]

IPYNBS = [e for e in files_from_folder(DOCS, regex='**/*.ipynb', skip=SKIP) if ".ipynb_checkpoints" not in e]


@pytest.mark.slow
@pytest.mark.parametrize('ipynb', IPYNBS)
def test_usage(ipynb):
    run_ipynb(ipynb, overwrite=OVERWRITE)
    assert True
