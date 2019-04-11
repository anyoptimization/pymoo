import os
import unittest

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor


class UsageTest(unittest.TestCase):

    def test(self):

        PYMOO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
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
                run_path = os.path.dirname(fname)
                ep.preprocess(nb, {'metadata': {'path': run_path}})

            except CellExecutionError:
                msg = 'Error executing the fname "%s".\n\n' % fname
                msg += 'See fname "%s" for the traceback.' % (fname + ".error")
                print(msg)
                raise

            #finally:
            #    with open((fname + ".error"), mode='wt') as f:
            #        nbformat.write(nb, f)


if __name__ == '__main__':
    unittest.main()
