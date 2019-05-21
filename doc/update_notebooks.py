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


# Update and run all notebooks
def update_and_run_notebook(filename, execute=False):
    # Try and open the .ipynb file
    try:
        nb = nbformat.read(filename, nbformat.NO_CONVERT)
    except FileNotFoundError:
        print(f"{filename} Not found in current directory")
        return

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
    while len(nb["cells"]) > 0 and nb["cells"][-1]['cell_type'] == 'code' and len(nb["cells"][-1]['source']) == 0\
            and nb["cells"][-1]['execution_count'] is None:
        nb["cells"] = nb["cells"][:-1]

    with open(filename, 'wt') as f:
        nbformat.write(nb, f)

    if execute:
        ep = ExecutePreprocessor(kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': get_pymoo()}})

    with open(filename, 'wt') as f:
        nbformat.write(nb, f)


if __name__ == "__main__":

    #files = glob.glob('source/problems/*.ipynb')
    files = glob.glob('source/algorithms/nsga2.ipynb')

    # files = ['source/problems/zdt.ipynb','source/problems/single.ipynb']

    for fname in files:
        print(fname)
        update_and_run_notebook(fname, execute=True)
