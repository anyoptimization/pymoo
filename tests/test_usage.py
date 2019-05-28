import os
import sys
import unittest
from subprocess import run, PIPE

from pymoo.configuration import get_pymoo


class UsageTest(unittest.TestCase):

    def test(self):

        USAGE_DIR = os.path.join(get_pymoo(), "pymoo", "usage", "visualization")
        SCRIPT_NAME = "test.py"
        usages = os.listdir(USAGE_DIR)
        print(usages)

        for fname in usages:

            with self.subTest(msg="Checking %s" % fname, fname=fname):

                if fname == "__init__.py":
                    continue

                # if fname != "usage_heatmap.py":
                #    continue

                if fname == "usage_heatmap.py":
                    print("test")

                print(fname)

                with open(os.path.join(USAGE_DIR, fname)) as f:
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


if __name__ == '__main__':
    unittest.main()
