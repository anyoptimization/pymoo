import json
import os
import unittest
from unittest.mock import MagicMock

import numpy as np

from pymoo.configuration import Configuration
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymop.problems.zdt import ZDT4


class CrossoverTest(unittest.TestCase):

    def test_sbx(self):

        with open(os.path.join("resources", "crossover.json"), encoding='utf-8') as f:
            data = json.loads(f.read())

        for i, e in enumerate(data):

            Configuration.rand.random = MagicMock()
            Configuration.rand.random.side_effect = e['rnd']

            sbx = SimulatedBinaryCrossover(0.9, 15)

            parents = np.array(e['parents'])[None, :, :]
            children = sbx.do(ZDT4(), parents)

            _children = np.array(e['children'])

            is_equal = np.all(np.abs(children - _children) < 0.001)

            if not is_equal:
                print(i)
                print(np.abs(children - _children))

            self.assertTrue(is_equal)


if __name__ == '__main__':
    unittest.main()
