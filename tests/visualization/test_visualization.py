import os
import unittest

from pymoo.configuration import get_pymoo
from tests.test_usage import test_usage


class VisualizationTest(unittest.TestCase):

    def test_usages(self):
        folder = os.path.join(get_pymoo(), "pymoo", "usage", "visualization")
        test_usage([os.path.join(folder, fname) for fname in os.listdir(folder) if fname.endswith("py")])

if __name__ == '__main__':
    unittest.main()
