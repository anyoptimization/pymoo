import os

import numpy as np

from pymoo.core.problem import Problem
from pymoo.util.matlab_engine import install_matlab, MatlabEngine

try:
    import matlab.engine
except:
    print(install_matlab())


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=10,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=0,
                         xu=1,
                         vtype=float)

        self.engine = None

    def _evaluate(self, X, out, *args, **kwargs):

        # if the matlab engine has not been started yet do this now
        if self.engine is None:
            # prepare the folder where the matlab files are
            folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code')

            # get the engine ready to start
            eng = MatlabEngine.get_instance()

            # change directory to be able to execute the files
            eng.cd(folder, nargout=0)

            # this can be used to initialize variables or load data if necessary - only called once
            eng._initialize(nargout=0)

            # store it in the object
            self.engine = eng

        eng = self.engine

        # convert the input to a matlab readable format
        mat_X = matlab.double(X.tolist())

        # execute the matlab function and take what is returned
        if not self.has_constraints():
            mat_F = eng.evaluate(mat_X, nargout=1)
            out["F"] = np.array(mat_F)
        else:
            mat_F, mat_G = eng.evaluate(mat_X, nargout=2)
            out["F"], out["G"] = np.array(mat_F), np.array(mat_G)


if __name__ == '__main__':
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.visualization.scatter import Scatter

    problem = MyProblem()

    algorithm = NSGA2(pop_size=100)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 200),
                   seed=1,
                   verbose=True)

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, color="red")
    plot.show()
