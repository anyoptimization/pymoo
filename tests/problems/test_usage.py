import os
import unittest


class UsageTest(unittest.TestCase):

    def test(self):

        USAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "pymop", "usage")

        print(USAGE_DIR)
        for fname in os.listdir(USAGE_DIR):

            with open(os.path.join(USAGE_DIR, fname)) as f:
                s = f.read()

                try:
                    exec(s, globals())
                except Exception as e:
                    print(fname)
                    print(e)
                    raise Exception("Usage %s failed." % fname)


if __name__ == '__main__':
    unittest.main()
