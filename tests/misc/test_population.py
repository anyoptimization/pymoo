import unittest

from pymoo.model.population import Population


class PopulationTest(unittest.TestCase):

    def test_has_method(self):
        pop = Population(100)
        self.assertTrue(pop.has("X"))

        self.assertFalse(pop.has("opt"))

        pop[:-1].set("opt", False)
        self.assertFalse(pop.has("opt"))

        pop[-1].set("opt", True)
        self.assertTrue(pop.has("opt"))


if __name__ == '__main__':
    unittest.main()
