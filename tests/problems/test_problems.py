import os
import unittest

from pymoo.configuration import get_pymoo
from tests.test_usage import test_usage


class ProblemsTest(unittest.TestCase):

    def test_problems(self):

        folder = os.path.join(get_pymoo(), "pymoo", "usage", "problems")
        files = [os.path.join(folder, fname) for fname in os.listdir(folder) if fname.endswith(".py")]

        files.append(os.path.join(get_pymoo(), "pymoo", "usage", "usage_problem.py"))

        test_usage(files)


if __name__ == '__main__':
    unittest.main()
