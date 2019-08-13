import os
import unittest

from pymoo.configuration import get_pymoo
from tests.test_usage import test_usage


class TerminationCriterionTest(unittest.TestCase):

    def test_termination_criterion(self):
        folder = os.path.join(get_pymoo(), "pymoo", "usage", "termination")
        test_usage([os.path.join(folder, fname) for fname in os.listdir(folder)])


if __name__ == '__main__':
    unittest.main()
