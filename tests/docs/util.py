import os
from pathlib import Path

import nbformat
from jupyter_client.manager import start_new_kernel
from nbconvert.preprocessors import CellExecutionError
from nbconvert.preprocessors.execute import executenb
from nbformat import read as nbread

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

EXAMPLES = os.path.join(ROOT, "examples")

TESTS = os.path.join(ROOT, "tests")

NO_INIT = ["__init__.py"]


def files_from_folder(folder, regex='**/*.py', skip=[]):
    files = [os.path.join(folder, fname) for fname in Path(folder).glob(regex)]
    return filter_by_exclude(files, exclude=skip)


def filter_by_exclude(files, exclude=[]):
    return [f for f in files if not any([os.path.basename(f) == s for s in exclude])]


def run_file(f):
    fname = os.path.basename(f)

    print("RUNNING:", fname)

    with open(f) as f:
        s = f.read()

        no_plots = "import matplotlib\n" \
                   "matplotlib.use('Agg')\n" \
                   "import matplotlib.pyplot as plt\n"

        s = no_plots + s + "\nplt.close()\n"

        exec(s, globals())


def run_ipynb(kernel, fname, overwrite=False, remove_trailing_empty_cells=False):
    print(fname.split("/")[-1])

    import warnings
    warnings.filterwarnings("ignore")

    config = dict(metadata=dict(path=os.path.dirname(fname)))

    try:
        nb = nbread(fname, as_version=4)
        executenb(nb, kernel=kernel, cwd=os.path.dirname(fname))

    except CellExecutionError as ex:
        msg = 'Error executing the fname "%s".\n\n' % fname
        print(msg)
        raise ex
    finally:
        if overwrite:

            if remove_trailing_empty_cells:
                while len(nb["cells"]) > 0 and nb["cells"][-1]['cell_type'] == 'code' and len(
                        nb["cells"][-1]['source']) == 0 \
                        and nb["cells"][-1]['execution_count'] is None:
                    nb["cells"] = nb["cells"][:-1]

            with open(fname, mode='wt') as f:
                nbformat.write(nb, f)


def remove_trailing_empty_cells(ipynb):
    nb = nbformat.read(ipynb, nbformat.NO_CONVERT)

    while len(nb["cells"]) > 0 and nb["cells"][-1]['cell_type'] == 'code' and len(
            nb["cells"][-1]['source']) == 0 \
            and nb["cells"][-1]['execution_count'] is None:
        nb["cells"] = nb["cells"][:-1]

    with open(ipynb, 'wt') as f:
        nbformat.write(nb, f)
