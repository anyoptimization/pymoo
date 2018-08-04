import numpy as np

from pymoo.algorithms.moead import MOEAD
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.algorithms.so_DE import DifferentialEvolution
from pymoo.algorithms.so_genetic_algorithm import SingleObjectiveGeneticAlgorithm
from pymoo.model.evaluator import Evaluator
from pymop.problem import Problem


def minimize(fun, xl=None, xu=None, termination=('n_eval', 10000), n_var=None, fun_args={}, method='auto', method_args={},
             seed=None, callback=None, disp=True):
    """

    Minimization of function of one or more variables, objectives and constraints.

    This is used as a convenience function to execute several algorithms with default settings which turned
    out to work for a test problems. However, evolutionary computations utilizes the idea of customizing a
    meta-algorithm. Customizing the algorithm using the object oriented interface is recommended to improve the
    convergence.

    Parameters
    ----------

    fun : callable
        A function that gets X which is a 2d array as input. Each row represents a solution to evaluate
        and each column a variable. The function needs to return also the same number of rows and each column
        is one objective to optimize. In case of constraints, a second 2d array can be returned.

    xl : numpy.array or number
        The lower boundaries of variables as a 1d array

    xu : numpy.array or number
        The upper boundaries of variables as a 1d array

    n_var : int
        If xl or xu is only a number, this is used to create the boundary numpy arrays. Other it remains unused.

    termination : tuple
        The termination criterium that is used to stop the algorithm when the result is satisfying.

    fun_args : dict
        Additional arguments if necessary to evaluate the solutions (constants, random seed, ...)


    method : string
        Algorithm that is used to solve the problem.

    callback : callable, optional
        Called after each iteration, as ``callback(D)``, where ``D`` is a dictionary with
        algorithm information, such as the current design, objective and constraint space.

    disp : bool
        Whether to display each generation the current result or not.

    Returns
    -------
    res : Result
        The optimization result represented as a ``Result`` object.


    """

    problem = None

    if isinstance(fun, Problem):
        problem = fun

    elif callable(fun):

        if xl is None or xu is None:
            raise Exception("Please provide lower and upper bounds for the problem.")
        if isinstance(xl, (int, float)):
            xl = xl * np.ones(n_var)
        if isinstance(xu, (int, float)):
            xu = xu * np.ones(n_var)

        # determine through a test evaluation details about the problem
        n_var = xl.shape[0]
        n_obj = -1
        n_constr = 0

        res = fun(xl[None, :], **fun_args)

        if isinstance(res, tuple):
            # if there are constraints it is a tuple of length two
            if len(res) > 1:
                n_constr = res[1].shape[1]
            n_obj = res[0].shape[1]
        else:
            n_obj = res.shape[1]

        # create an optimization problem given the function properties
        class OptimizationProblem(Problem):
            def __init__(self):
                Problem.__init__(self)
                self.n_var = n_var
                self.n_constr = n_constr
                self.n_obj = n_obj
                self.func = self._evaluate
                self.xl = xl
                self.xu = xu

            def _evaluate(self, x, f, g=None):
                if g is None:
                    f[:, :] = fun(x, **fun_args)
                else:
                    f[:, :], g[:, :] = fun(x, **fun_args)

        problem = OptimizationProblem()

    else:
        raise Exception("fun arguments needs to be either a function or an object of the class problem.")

    # create an evaluator defined by the termination criterium
    termination_criterium, termination_val = termination
    if termination_criterium == 'n_eval':
        evaluator = Evaluator(termination_val)
    else:
        raise Exception('Unknown Termination criterium: %s' % termination_criterium)


    return minimize_(problem, evaluator, method=method, method_args=method_args, seed=seed,
                     callback=callback,
                     disp=disp)


def minimize_(problem, evaluator, method='auto', method_args={}, seed=None,
              callback=None, disp=False):
    """
        See :func:`~pymoo.optimize.minimize` for description. Instead of a function the parameter is a problem class.
    """

    # try to find a good algorithm if auto is selected
    if method == 'auto':
        method = 'nsga2'

    # choose the algorithm implementation given the string
    if method == 'nsga2':
        algorithm = NSGA2("real", disp=disp, **method_args)
    elif method == 'nsga3':
        algorithm = NSGA3("real", disp=disp, **method_args)
    elif method == 'moead':
        algorithm = MOEAD("real", disp=disp, **method_args)
    elif method == 'de':
        algorithm = DifferentialEvolution(disp=disp, **method_args)
    elif method == 'ga':
        algorithm = SingleObjectiveGeneticAlgorithm("real", disp=disp, **method_args)
    else:
        raise Exception('Unknown method: %s' % method)

    X, F, G = algorithm.solve(problem, evaluator)

    return {'problem': problem, 'X': X, 'F': F, 'G': G}
