from jupyter_client.manager import start_new_kernel

from tests.test_util import files_from_folder, run_ipynb, DOCS

SKIP = ["parallelization.ipynb",
        "video.ipynb",
        "modact.ipynb",
        "dascmop.ipynb"]

IPYNBS = [e for e in files_from_folder(DOCS, regex='**/*.ipynb', skip=SKIP) if ".ipynb_checkpoints" not in e]

# IPYNBS = ['/Users/blankjul/workspace/pymoo/docs/source/algorithms/moo/nsga2.ipynb']

failed = []

for ipynb in IPYNBS:
    print(f"jupytext --to md:myst {ipynb}")
    if False:
        print(ipynb, end="")
        try:
            KERNEL = start_new_kernel(kernel_name='default')
            run_ipynb(KERNEL, ipynb, overwrite=True, remove_trailing_empty_cells=True, verbose=False)
            print(" OK")
        except:
            failed.append(ipynb)
            print(" FAIL")

print("FAILED:", len(failed))

for e in failed:
    print(e)
