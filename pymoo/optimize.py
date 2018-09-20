import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.algorithms.rnsga3 import RNSGA3
from pymoo.algorithms.so_genetic_algorithm import SingleObjectiveGeneticAlgorithm
from pymoo.algorithms.unsga3 import UNSGA3
from pymoo.model.termination import MaximumFunctionCallTermination, MaximumGenerationTermination, IGDTermination
from pymoo.rand import random
from pymop.problem import Problem


def minimize(fun,
             method,
             xl=None,
             xu=None,
             termination=('n_gen', 200),
             n_var=None,
             fun_args={},
             method_args={},
             seed=None,
             callback=None,
             disp=True,
             save_history=False):

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
    method_args : dict
        Additional arguments to initialize the algorithm object
    seed : int
        Random seed to be used for the run
    callback : callable, optional
        Called after each iteration, as ``callback(D)``, where ``D`` is a dictionary with
        algorithm information, such as the current design, objective and constraint space.
    disp : bool
        Whether to display each generation the current result or not.
    save_history : bool
        If true every iteration a snapshot of the algorithm is stored.

    Returns
    -------
    res : Result
        The optimization result represented as a ``Result`` object.


    """

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
        termination = MaximumFunctionCallTermination(termination_val)
    elif termination_criterium == 'n_gen':
        termination = MaximumGenerationTermination(termination_val)
    elif termination_criterium == 'igd':
        termination = IGDTermination(problem, termination_val)
    else:
        raise Exception('Unknown Termination criterium: %s' % termination_criterium)

    # set a random random seed if not provided
    if seed is None:
        seed = random.randint(1, 10000)

    return _minimize(problem, termination, method, method_args=method_args, seed=seed,
                     callback=callback, disp=disp, save_history=save_history)


def _minimize(problem, termination, method, method_args={}, seed=1,
              callback=None, disp=False, save_history=False):
    """
        See :func:`~pymoo.optimize.minimize` for description. Instead of a function the parameter is a problem class.
    """

    # choose the algorithm implementation given the string
    if method == 'nsga2':
        algorithm = NSGA2(**method_args)
    elif method == 'nsga3':
        algorithm = NSGA3(**method_args)
    elif method == 'rnsga3':
        algorithm = RNSGA3(**method_args)
    elif method == 'unsga3':
        algorithm = UNSGA3(**method_args)
    elif method == 'ga':
        algorithm = SingleObjectiveGeneticAlgorithm(**method_args)
    else:
        raise Exception('Unknown method: %s' % method)

    res = algorithm.solve(problem,
                          termination,
                          disp=disp,
                          callback=callback,
                          seed=seed,
                          return_only_feasible=False,
                          return_only_non_dominated=False,
                          save_history=save_history)

    return res
