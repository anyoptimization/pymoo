import os
import unittest


import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

from pymoo.configuration import get_pymoo

OVERWRITE = False

STARTING_AT = None

SKIP = ["parallelization.ipynb"]


class DocumentationTest(unittest.TestCase):

    def test(self):

        PYMOO_DIR = get_pymoo()
        DOC_DIR = os.path.join(PYMOO_DIR, "doc", "source")
        ipynb = []

        # collect all the jupyter ipynb in the documentation
        for root, directories, filenames in os.walk(DOC_DIR):
            for filename in filenames:
                if filename.endswith(".ipynb") and "checkpoint" not in filename and not any([filename in s for s in SKIP]):
                    ipynb.append(os.path.join(root, filename))

        i = 0
        if STARTING_AT is not None:
            for j in range(len(ipynb)):
                if STARTING_AT not in ipynb[j]:
                    i += 1
                else:
                    break

        ep = ExecutePreprocessor(timeout=10000, kernel_name='python3')

        for i in range(i, len(ipynb)):

            fname = ipynb[i]

            print(fname.split("/")[-1])

            import warnings
            warnings.filterwarnings("ignore")

            try:
                nb = nbformat.read(fname, nbformat.NO_CONVERT)
                ep.preprocess(nb, {'metadata': {'path': PYMOO_DIR}})

            except CellExecutionError:
                msg = 'Error executing the fname "%s".\n\n' % fname
                print(msg)
                raise
            finally:
                if OVERWRITE:
                    with open(fname, mode='wt') as f:
                        nbformat.write(nb, f)


if __name__ == '__main__':
    unittest.main()
