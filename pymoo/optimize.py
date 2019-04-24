from pymoo.factory import get_termination
from pymoo.model.termination import Termination
from pymoo.rand import random


def minimize(problem,
             method,
             termination,
             **kwargs):
    """

    Minimization of function of one or more variables, objectives and constraints.

    This is used as a convenience function to execute several algorithms with default settings which turned
    out to work for a test problems. However, evolutionary computations utilizes the idea of customizing a
    meta-algorithm. Customizing the algorithm using the object oriented interface is recommended to improve the
    convergence.

    Parameters
    ----------

    problem : pymop.problem
        A problem object defined using the pymop framework. Either existing test problems or custom problems
        can be provided. please have a look at the documentation.

    method : :class:`~pymoo.model.algorithm.Algorithm`
        The algorithm object that should be used for the optimization.

    termination : tuple
        The termination criterium that is used to stop the algorithm when the result is satisfying.

    Returns
    -------
    res : :class:`~pymoo.model.result.Result`
        The optimization result represented as an object.

    """

    # create an evaluator defined by the termination criterium
    if not isinstance(termination, Termination):
        termination = get_termination(*termination)

    # set a random random seed if not provided
    if 'seed' not in kwargs:
        kwargs['seed'] = random.randint(1, 10000)

    res = method.solve(problem, termination, **kwargs)

    return res
