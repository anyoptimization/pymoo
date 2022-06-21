import os

import pytest
from jupyter_client.manager import start_new_kernel

from util import files_from_folder, run_ipynb, ROOT

DOCS = os.path.join(ROOT, "source")

SKIP = ["parallelization.ipynb", "video.ipynb", "modact.ipynb", "dascmop.ipynb", "constraints.ipynb"]

IPYNBS = [e for e in files_from_folder(DOCS, regex='**/*.ipynb', skip=SKIP) if ".ipynb_checkpoints" not in e]

KERNEL = start_new_kernel(kernel_name='python3')

@pytest.mark.parametrize('ipynb', IPYNBS)
def test_docs(ipynb, pytestconfig):
    overwrite = pytestconfig.getoption("overwrite")
    KERNEL[0].restart_kernel(now=True)
    run_ipynb(KERNEL, ipynb, overwrite=overwrite, remove_trailing_empty_cells=True)
    assert True
