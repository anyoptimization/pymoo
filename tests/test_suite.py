import os
import sys
import unittest

if __name__ == "__main__":

    # add the path to be execute in the main directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    testmodules = [
        'tests.test_nsga2',
        'pymoo.usage.test_usage'
    ]

    suite = unittest.TestSuite()

    for t in testmodules:
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

    unittest.TextTestRunner().run(suite)