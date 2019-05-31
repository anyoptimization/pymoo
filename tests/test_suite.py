import os
import sys
import unittest

if __name__ == "__main__":

    # add the path to be execute in the main directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    testmodules = [
        'tests.test_nsga2',
        'tests.test_algorithms',
        'tests.test_problems',
        'tests.test_crossover',
        'tests.test_performance_indicator',
        'tests.test_visualization',
        'tests.problems.test_correctness',
        'tests.test_termination_criterion'
        #'tests.problems.test_gradient',
        #'tests.problems.test_hessian',
        #'tests.problems.test_usage'
        #'pymoo.usage.test_usage'
    ]

    suite = unittest.TestSuite()

    for t in testmodules:
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

    unittest.TextTestRunner().run(suite)