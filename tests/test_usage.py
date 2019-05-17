import os
import traceback
import unittest


class UsageTest(unittest.TestCase):

    def test(self):

        USAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "usage")
        # USAGE_DIR = os.path.dirname(os.path.realpath(__file__))

        usages = os.listdir(USAGE_DIR)
        print(usages)

        for fname in usages:

            with self.subTest(msg="Checking %s" % fname, fname=fname):

                print(fname + " -> OK", )

                with open(os.path.join(USAGE_DIR, fname)) as f:
                    s = f.read()

                    no_plots = "import matplotlib\n" \
                               "import matplotlib.pyplot\n" \
                               "matplotlib.use('Agg')\n"

                    s = no_plots + s

                    s += "\nmatplotlib.pyplot.close()\n"

                    try:
                        exec(s, globals())
                    except:
                        traceback.print_exc()
                        raise Exception("Usage %s failed." % fname)


if __name__ == '__main__':
    unittest.main()
