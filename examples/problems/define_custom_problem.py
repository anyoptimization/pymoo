import numpy as np

from pymoo.core.problem import Problem


# always derive from the main problem for the evaluation
class MyProblem(Problem):

    def __init__(self, const_1=5, const_2=0.1):

        # define lower and upper bounds -  1d array with length equal to number of variable
        xl = -5 * np.ones(10)
        xu = 5 * np.ones(10)

        super().__init__(n_var=10, n_obj=1, n_ieq_constr=2, xl=xl, xu=xu, evaluation_of="auto")

        # store custom variables needed for evaluation
        self.const_1 = const_1
        self.const_2 = const_2

    # implemented the function evaluation function - the arrays to fill are provided directly
    def _evaluate(self, x, out, *args, **kwargs):
        # define an objective function to be evaluated using var1
        f = np.sum(np.power(x, 2) - self.const_1 * np.cos(2 * np.pi * x), axis=1)

        # !!! only if a constraint value is positive it is violated !!!
        # set the constraint that x1 + x2 > var2
        g1 = (x[:, 0] + x[:, 1]) - self.const_2

        # set the constraint that x3 + x4 < var2
        g2 = self.const_2 - (x[:, 2] + x[:, 3])

        out["F"] = f
        out["G"] = np.column_stack([g1, g2])


problem = MyProblem()

X = np.random.rand(100, 10)
F, G, dF, dG = problem.evaluate(X, return_values_of=["F", "G", "dF", "dG"])
print(F, G, dF, dG)
