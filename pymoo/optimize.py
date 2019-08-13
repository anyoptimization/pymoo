import copy

from pymoo.factory import get_termination
from pymoo.model.termination import Termination


def minimize(problem,
             algorithm,
             termination=None,
             **kwargs):
    """

    Minimization of function of one or more variables, objectives and constraints.

    This is used as a convenience function to execute several algorithms with default settings which turned
    out to work for a test single. However, evolutionary computations utilizes the idea of customizing a
    meta-algorithm. Customizing the algorithm using the object oriented interface is recommended to improve the
    convergence.

    Parameters
    ----------

    problem : pymoo.problem
        A problem object which is defined using pymoo.

    algorithm : :class:`~pymoo.model.algorithm.Algorithm`
        The algorithm object that should be used for the optimization.

    termination : class or tuple
        The termination criterion that is used to stop the algorithm.


    Returns
    -------
    res : :class:`~pymoo.model.result.Result`
        The optimization result represented as an object.

    """

    # create a copy of the algorithm object to ensure no side-effects
    algorithm = copy.deepcopy(algorithm)

    # set the termination criterion and store it in the algorithm object
    if termination is None:
        termination = None
    elif not isinstance(termination, Termination):
        if isinstance(termination, str):
            termination = get_termination(termination)
        else:
            termination = get_termination(*termination)

    # initialize the method given a problem
    algorithm.initialize(problem,
                         termination=termination,
                         **kwargs)

    # actually execute the algorithm
    res = algorithm.solve()

    # store the copied algorithm in the result object
    res.algorithm = algorithm

    return res
