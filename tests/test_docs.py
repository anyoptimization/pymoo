import pytest
from jupyter_client.manager import start_new_kernel

from tests.test_util import files_from_folder, run_ipynb, DOCS

SKIP = ["parallelization.ipynb",
        "video.ipynb",
        "modact.ipynb",
        "dascmop.ipynb",
        "constraints.ipynb"]

IPYNBS = [e for e in files_from_folder(DOCS, regex='**/*.ipynb', skip=SKIP) if ".ipynb_checkpoints" not in e]


@pytest.mark.long
@pytest.mark.parametrize('ipynb', IPYNBS)
def test_docs(ipynb, pytestconfig):
    overwrite = pytestconfig.getoption("overwrite", False)
    KERNEL = start_new_kernel(kernel_name='python3')
    run_ipynb(KERNEL, ipynb, overwrite=overwrite, remove_trailing_empty_cells=True)
    assert True
