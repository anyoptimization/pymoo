import unittest

from pymoo.factory import get_problem_options, get_sampling, get_problem
from pymoo.model.population import Population
from pymoo.vendor.global_opt import get_global_optimization_problem_options


exclude = [
"knp"

]


class NoErrorRandomSamplingProblemTest(unittest.TestCase):

    def test_problems(self):

        PROBLEMS = get_problem_options()
        PROBLEMS.extend(get_global_optimization_problem_options())

        for _tuple in PROBLEMS:
            name = _tuple[0]
            print(name, "OK")

            if name in exclude:
                continue

            problem = get_problem(name)

            X = get_sampling("real_random").do(problem, Population(), 100).get("X")

            out = problem.evaluate(X)



if __name__ == '__main__':
    unittest.main()
