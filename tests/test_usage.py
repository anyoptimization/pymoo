import os
import unittest


class UsageTest(unittest.TestCase):

    def test(self):

        USAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "pymoo", "usage")
        # USAGE_DIR = os.path.dirname(os.path.realpath(__file__))

        usages = os.listdir(USAGE_DIR)
        print(usages)

        for fname in usages:

            if fname == "test_usage.py":
                continue

            with self.subTest(msg="Checking %s" % fname, fname=fname):

                print("=" * 50)
                print(fname)
                print("=" * 50)

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
                        raise Exception("Usage %s failed." % fname)


if __name__ == '__main__':
    unittest.main()
