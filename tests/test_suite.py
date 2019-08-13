import os
import sys
import unittest

if __name__ == "__main__":

    # add the path to be execute in the main directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    testmodules = [
        'tests.algorithms.test_nsga2',
        'tests.algorithms.test_algorithms',
        'tests.problems.test_problems',
        'tests.operators.test_crossover',
        'tests.performance_indicator.test_performance_indicator',
        'tests.visualization.test_visualization',
        'tests.problems.test_correctness',
        'tests.termination_criterion.test_termination_criterion'
    ]

    suite = unittest.TestSuite()

    for t in testmodules:
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

    unittest.TextTestRunner().run(suite)