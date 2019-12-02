import os
import sys
import unittest
from pathlib import Path


DISABLED = ['test_docs']

if __name__ == "__main__":

    # add the path to be execute in the main directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    suite = unittest.TestSuite()
    
    for test in Path('.').rglob('test_*.py'):

        name = str(test)[:-3].replace('/', '.')

        if name not in DISABLED:

            print(name)

            entry = unittest.defaultTestLoader.loadTestsFromName(name)
            suite.addTest(entry)


    ret = unittest.TextTestRunner().run(suite)

    if len(ret.failures) + len(ret.errors) > 0:
        exit(1)