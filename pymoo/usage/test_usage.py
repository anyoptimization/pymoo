import os
import unittest

class UsageTest(unittest.TestCase):

    def test(self):

        #USAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "pymoo", "usage")
        USAGE_DIR = os.path.dirname(os.path.realpath(__file__))

        for fname in os.listdir(USAGE_DIR):

            if fname == "test_usage.py":
                continue

            with self.subTest(msg="Checking %s" % fname, fname=fname):

                with open(os.path.join(USAGE_DIR, fname)) as f:
                    s = f.read()

                    no_plots = "from matplotlib import pyplot as plt\n" \
                                    "plt.ioff()\n"
                    s = no_plots + s

                    try:
                        exec(s, globals())
                    except:
                        raise Exception("Usage %s failed." % fname)


if __name__ == '__main__':
    unittest.main()
