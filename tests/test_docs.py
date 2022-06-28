import os
from os.path import dirname

import pytest
from jupyter_client.manager import start_new_kernel

from pymoo.config import Config
from tests.test_util import files_from_folder, run_ipynb

FOLDER = os.path.join(dirname(Config.root), "docs", "source")

SKIP = ["parallelization.ipynb",
        "video.ipynb",
        "modact.ipynb",
        "dascmop.ipynb",
        "constraints.ipynb"]

IPYNBS = [e for e in files_from_folder(FOLDER, regex='**/*.ipynb', skip=SKIP) if ".ipynb_checkpoints" not in e]

KERNEL = start_new_kernel(kernel_name='python3')


@pytest.mark.long
@pytest.mark.parametrize('ipynb', IPYNBS)
def test_docs(ipynb, pytestconfig):
    overwrite = pytestconfig.getoption("overwrite", False)
    KERNEL[0].restart_kernel(now=True)
    run_ipynb(KERNEL, ipynb, overwrite=overwrite, remove_trailing_empty_cells=True)
    assert True
