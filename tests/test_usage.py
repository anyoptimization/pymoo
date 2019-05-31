import os
import sys
import unittest
from pathlib import Path
from subprocess import run, PIPE

from pymoo.configuration import get_pymoo


def test_usage(usages):
    SCRIPT_NAME = "test.py"
    usages = [f for f in usages if f != "__init__.py"]

    print(usages)

    for path_to_file in usages:

        fname = os.path.basename(path_to_file)

        if fname == "__init__.py":
            continue

        print(fname)

        with open(path_to_file) as f:
            s = f.read()

            no_plots = "import matplotlib\n" \
                       "import matplotlib.pyplot\n" \
                       "matplotlib.use('Agg')\n"

            s = no_plots + s + "\nmatplotlib.pyplot.close()\n"

            if os.path.exists(SCRIPT_NAME):
                os.remove(SCRIPT_NAME)

            with open(SCRIPT_NAME, 'a') as out:
                out.write(s)

            command = ["python", "test.py"]
            result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
            os.remove(SCRIPT_NAME)

            if result.returncode == 1:
                print(result.stderr, file=sys.stderr)
                raise Exception("Error")
            else:
                print("OK\n")


class AllUsageTest(unittest.TestCase):

    def test_usages(self):
        folder = os.path.join(get_pymoo(), "pymoo", "usage")
        test_usage([os.path.join(folder, fname) for fname in Path(folder).glob('**/*.py')])



if __name__ == '__main__':
    unittest.main()
