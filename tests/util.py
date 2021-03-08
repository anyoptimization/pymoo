import os
from pathlib import Path

from pymoo.configuration import get_pymoo

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(ROOT)

USAGE = os.path.join(ROOT, "pymoo", "usage")

TESTS = os.path.join(ROOT, "tests")

RESOURCES = os.path.join(TESTS, "resources")
print(RESOURCES)


def path_to_test_resources(*args):
    return os.path.join(RESOURCES, *args)


NO_INIT = ["__init__.py"]


def files_from_folder(folder, regex='**/*.py', skip=[]):
    files = [os.path.join(folder, fname) for fname in Path(folder).glob(regex)]
    return filter_by_exclude(files, exclude=skip)


def filter_by_exclude(files, exclude=[]):
    return [f for f in files if not any([os.path.basename(f) == s for s in exclude])]


def run_usage(f):
    fname = os.path.basename(f)

    print("RUNNING:", fname)

    with open(f) as f:
        s = f.read()

        no_plots = "import matplotlib\n" \
                   "matplotlib.use('Agg')\n" \
                   "import matplotlib.pyplot as plt\n"

        s = no_plots + s + "\nplt.close()\n"

        exec(s, globals())


def run_ipynb(fname, overwrite=False):
    import nbformat
    from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

    ep = ExecutePreprocessor(timeout=10000, kernel_name='python3')

    print(fname.split("/")[-1])

    import warnings
    warnings.filterwarnings("ignore")

    try:
        nb = nbformat.read(fname, nbformat.NO_CONVERT)
        ep.preprocess(nb, {'metadata': {'path': get_pymoo()}})

    except CellExecutionError as ex:
        msg = 'Error executing the fname "%s".\n\n' % fname
        print(msg)
        raise ex
    finally:
        if overwrite:
            with open(fname, mode='wt') as f:
                nbformat.write(nb, f)
