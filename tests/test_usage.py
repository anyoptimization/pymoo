import os
import unittest
from pathlib import Path

from pymoo.configuration import get_pymoo

SKIP = ["__init__.py", "usage_video.py", "usage_stream.py", "usage_mopta08.py", "__pycache__"]


def test_usage(usages):
    usages = [f for f in usages if not any([f.endswith(s) for s in SKIP])]

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

            try:
                exec(s, globals())
            except:
                raise Exception("Usage %s failed." % fname)


class AllUsageTest(unittest.TestCase):

    def test_usages(self):
        folder = os.path.join(get_pymoo(), "pymoo", "usage")
        test_usage([os.path.join(folder, fname) for fname in Path(folder).glob('**/*.py')])


if __name__ == '__main__':
    unittest.main()
