import unittest

import numpy as np
from distributed import Client
import os

from pymoo.model.problem import Problem

#client = Client(address="host-94108.dhcp.egr.msu.edu:8786")
client = Client(address="localhost:9000")
client.upload_file(os.path.realpath(__file__))




class ParallelEvaluationTest(unittest.TestCase):

    def get_data(self):
        np.random.seed(1)
        X = np.random.random((100, 2))
        F = X.sum(axis=1)[:, None]
        return X, F

    def test_evaluation_in_threads(self):
        X, F = self.get_data()
        _F = MyProblemElementwise(parallelization="threads").evaluate(X)
        self.assertTrue(np.all(np.abs(_F - F) < 0.00001))

    def test_evaluation_in_threads_number(self):
        X, F = self.get_data()
        _F = MyProblemElementwise(parallelization=("threads", 2)).evaluate(X)
        self.assertTrue(np.all(np.abs(_F - F) < 0.00001))


    def test_evaluation_with_dask(self):
        X, F = self.get_data()

        p = Problem(n_var=2, n_obj=1, elementwise_evaluation=True, parallelization=("dask", client))
        p._evaluate = MyProblemElementwise._evaluate


        _F = p.evaluate(X)
        self.assertTrue(np.all(np.abs(_F - F) < 0.00001))


class MyProblemElementwise(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, elementwise_evaluation=True, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x.sum()


if __name__ == '__main__':
    unittest.main()
