import glob

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from pymoo.configuration import get_pymoo

usage = {}


def get_usage(fname, prefix="../pymoo/usage/"):
    if fname not in usage:
        with open(prefix + fname, 'r') as f:
            usage[fname] = f.read()
    return usage[fname]


def get_section(code, section):
    start, end = None, None
    code = code.split("\n")

    for i, line in enumerate(code):

        if "# START %s" % section in line:
            start = i
        elif "# END %s" % section in line:
            end = i
            break

    if start is None:
        raise Exception("%s not found." % section)

    return "\n".join(code[start + 1:end]).rstrip().lstrip()


def load_notebook(filename):
    try:
        return nbformat.read(filename, nbformat.NO_CONVERT)
    except FileNotFoundError:
        raise Exception(f"{filename} Not found in current directory")


def replace_usage_link(filename, old, new):
    nb = load_notebook(filename)
    for k in range(len(nb['cells'])):

        if nb['cells'][k]['metadata'].get('code') == old:
            nb['cells'][k]['metadata'].set('code', new)

    with open(filename, 'wt') as f:
        nbformat.write(nb, f)


def update_and_run_notebook(filename, replace=True, execute=False):
    nb = load_notebook(filename)

    if replace:

        # Loop through all cells
        for k in range(len(nb['cells'])):
            # If cell is code and has usage metadata set, update the .ipynb json
            if nb['cells'][k]['cell_type'] == 'code':

                code = nb['cells'][k]['metadata'].get('code')
                section = nb['cells'][k]['metadata'].get('section')

                if code:
                    code = get_usage(code)

                    if section:
                        code = get_section(code, section)

                    nb['cells'][k]['source'] = code

        # remove trailing empty cells
        while len(nb["cells"]) > 0 and nb["cells"][-1]['cell_type'] == 'code' and len(nb["cells"][-1]['source']) == 0 \
                and nb["cells"][-1]['execution_count'] is None:
            nb["cells"] = nb["cells"][:-1]

    if execute:
        ep = ExecutePreprocessor(kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': get_pymoo()}})

    with open(filename, 'wt') as f:
        nbformat.write(nb, f)


if __name__ == "__main__":

    # files = glob.glob('source/**/*.ipynb')
    # files = glob.glob('source/problems/*.ipynb')
    files = glob.glob('source/algorithms/pso.ipynb')
    # files = glob.glob('source/visualization/video.ipynb')
    # files = glob.glob('source/visualization/*.ipynb')
    # files = glob.glob('source/components/performance_indicator.ipynb')
    # files = glob.glob('source/misc/termination_criterion.ipynb')

    # STARTING_AT = "source/customization/initialization.ipynb"
    #STARTING_AT = "source/misc/results.ipynb"
    STARTING_AT = None

    SKIP = ["source/problems/parallelization.ipynb", "source/visualization/video.ipynb"]

    #files = ['source/misc/custom_output.ipynb']

    for fname in files:

        if fname in SKIP:
            continue

        if STARTING_AT is not None and fname in STARTING_AT:
            STARTING_AT = None

        if STARTING_AT is None:
            print(fname)
            update_and_run_notebook(fname, execute=True)
