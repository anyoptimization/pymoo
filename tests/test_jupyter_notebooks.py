import os
import unittest

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

from pymoo.configuration import get_pymoo

OVERWRITE = True


class UsageTest(unittest.TestCase):

    def test(self):

        PYMOO_DIR = get_pymoo()
        DOC_DIR = os.path.join(PYMOO_DIR, "doc", "source")
        ipynb = []

        # collect all the jupyter ipynb in the documentation
        for root, directories, filenames in os.walk(DOC_DIR):
            for filename in filenames:
                if filename.endswith(".ipynb") and "checkpoint" not in filename:
                    ipynb.append(os.path.join(root, filename))

        ep = ExecutePreprocessor(timeout=10000, kernel_name='python3')

        for fname in ipynb:

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
