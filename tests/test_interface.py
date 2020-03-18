import unittest

import numpy as np

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.interface import AskAndTell


class InterfaceTest(unittest.TestCase):

    def test_ask_and_tell(self):

        # set the random seed for this test
        np.random.seed(1)

        # create the algorithm object to be used
        algorithm = GA(pop_size=100, eliminate_duplicates=True)

        # create the ask and tell interface object
        api = AskAndTell(algorithm, n_var=2, n_obj=1, n_constr=1, xl=-10, xu=10)

        # this loop runs always one step of the algorithm
        for gen in range(200):

            # ask the algorithm for values to be evaluated
            X = api.ask()

            # evaluate the values - here just some easy calculations
            F = np.sum(np.square(X), axis=1)[:, None]
            G = 1 - np.abs(X[:, 0])[:, None]

            # let the api objects know the objective and constraint values
            api.tell(F, G=G)

            print(api.get_population().get("F").min())

        # retrieve the results form the api - here the whole population of the algorithm
        X, F, CV = api.result(only_optimum=False, return_values_of=["X", "F", "CV"])

        self.assertTrue(np.allclose(CV, 0))
        self.assertTrue(np.allclose(F[:10], 1, atol=1.e-3))




if __name__ == '__main__':
    unittest.main()
