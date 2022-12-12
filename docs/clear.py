import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

fname = sys.argv[1]

with open(fname) as f:
    nb = nbformat.read(f, as_version=4)

nb["metadata"] = {}

cells = []
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cells.append(cell)

nb["cells"] = cells

with open(fname, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)