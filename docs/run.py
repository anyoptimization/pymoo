from jupyter_client.manager import start_new_kernel

from tests.test_util import files_from_folder, run_ipynb, DOCS

SKIP = ["parallelization.ipynb",
        "video.ipynb",
        "modact.ipynb",
        "dascmop.ipynb"]

IPYNBS = [e for e in files_from_folder(DOCS, regex='**/*.ipynb', skip=SKIP) if ".ipynb_checkpoints" not in e]

failed = []

for ipynb in IPYNBS:
    print(ipynb, end="")
    try:
        KERNEL = start_new_kernel(kernel_name='python3')
        run_ipynb(KERNEL, ipynb, overwrite=True, remove_trailing_empty_cells=True, verbose=False)
        print(" OK")
    except:
        failed.append(ipynb)
        print(" FAIL")

print("FAILED:", len(failed))

for e in failed:
    print(e)
