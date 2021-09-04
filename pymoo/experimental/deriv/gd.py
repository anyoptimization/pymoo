import numpy as np

from pymoo.experimental.deriv import DerivationBasedAlgorithm
from pymoo.algorithms.base.line import LineSearchProblem
from pymoo.algorithms.soo.univariate.exp import ExponentialSearch
from pymoo.algorithms.soo.univariate.golden import GoldenSectionSearch
from pymoo.core.population import Population
from pymoo.util.vectors import max_alpha


class GradientDescent(DerivationBasedAlgorithm):

    def direction(self, dF, **kwargs):
        return - dF

    def step(self):
        problem, sol = self.problem, self.opt[0]
        self.evaluator.eval(self.problem, sol, evaluate_values_of=["dF"])
        dF = sol.get("dF")[0]

        print(sol)

        if np.linalg.norm(dF) ** 2 < 1e-8:
            self.termination.force_termination = True
            return

        direction = self.direction(dF)

        line = LineSearchProblem(self.problem, sol, direction, strict_bounds=self.strict_bounds)
        alpha = self.alpha

        if self.strict_bounds:

            if problem.has_bounds():
                line.xu = np.array([max_alpha(sol.X, direction, *problem.bounds(), mode="all_hit_bounds")])

            # remember the step length from the last run
            alpha = min(alpha, line.xu[0])

            if alpha == 0:
                self.termination.force_termination = True
                return

        # make the solution to be the starting point of the univariate search
        x0 = sol.copy(deep=True)
        x0.set("__X__", x0.get("X"))
        x0.set("X", np.zeros(1))

        # determine the brackets to be searched in
        exp = ExponentialSearch(delta=alpha).setup(line, evaluator=self.evaluator, termination=("n_iter", 20), x0=x0)
        a, b = exp.run().pop[-2:]

        # search in the brackets
        res = GoldenSectionSearch().setup(line, evaluator=self.evaluator, termination=("n_iter", 20), a=a, b=b).run()
        infill = res.opt[0]

        # set the alpha value and revert the X to be the multi-variate one
        infill.set("X", infill.get("__X__"))
        self.alpha = infill.get("alpha")[0]

        # keep always a few historical solutions
        self.pop = Population.merge(self.pop, infill)[-10:]
