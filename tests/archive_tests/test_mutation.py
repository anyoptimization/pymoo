import json
import os
import unittest
from unittest.mock import MagicMock

import numpy as np

from pymoo.configuration import Configuration
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymop.problems.zdt import ZDT4


class CrossoverTest(unittest.TestCase):

    def test_pm(self):

        with open(os.path.join("resources", "mutation.json"), encoding='utf-8') as f:
            data = json.loads(f.read())

        for i, e in enumerate(data):

            Configuration.rand.random = MagicMock()
            Configuration.rand.random.side_effect = e['rnd']

            pm = PolynomialMutation(eta_mut=20, prob_mut=0.1)

            parents = np.array(e['ind'])
            children = pm.do(ZDT4(), parents)

            _children = np.array(e['off'])

            is_equal = np.all(np.abs(children - _children) < 0.1)

            if not is_equal:
                print(i)
                print(np.abs(children - _children))

            self.assertTrue(is_equal)


if __name__ == '__main__':
    unittest.main()
