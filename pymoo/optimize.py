import copy

from pymoo.factory import get_termination
from pymoo.model.termination import Termination
from pymoo.util.misc import termination_from_tuple
from pymoo.util.termination.default import MultiObjectiveDefaultTermination, SingleObjectiveDefaultTermination


def minimize(problem,
             algorithm,
             termination=None,
             copy_algorithm=True,
             copy_termination=True,
             **kwargs):
    """

    Minimization of function of one or more variables, objectives and constraints.

    This is used as a convenience function to execute several algorithms with default settings which turned
    out to work for a test single. However, evolutionary computations utilizes the idea of customizing a
    meta-algorithm. Customizing the algorithm using the object oriented interface is recommended to improve the
    convergence.

    Parameters
    ----------

    problem : :class:`~pymoo.model.problem.Problem`
        A problem object which is defined using pymoo.

    algorithm : :class:`~pymoo.model.algorithm.Algorithm`
        The algorithm object that should be used for the optimization.

    termination : :class:`~pymoo.model.termination.Termination` or tuple
        The termination criterion that is used to stop the algorithm.

    seed : integer
        The random seed to be used.

    verbose : bool
        Whether output should be printed or not.

    display : :class:`~pymoo.util.display.Display`
        Each algorithm has a default display object for printouts. However, it can be overwritten if desired.

    callback : :class:`~pymoo.model.callback.Callback`
        A callback object which is called each iteration of the algorithm.

    save_history : bool
        Whether the history should be stored or not.

    copy_algorithm : bool
        Whether the algorithm object should be copied before optimization.

    copy_termination : bool
        Whether the termination object should be copied before called.

    Returns
    -------
    res : :class:`~pymoo.model.result.Result`
        The optimization result represented as an object.

    """

    # create a copy of the algorithm object to ensure no side-effects
    if copy_algorithm:
        algorithm = copy.deepcopy(algorithm)

    # initialize the algorithm object given a problem - if not set already
    if algorithm.problem is None:
        algorithm.setup(problem, termination=termination, **kwargs)
    else:
        if termination is not None:
            algorithm.termination = termination_from_tuple(termination)

    # if no termination could be found add the default termination either for single or multi objective
    termination = algorithm.termination
    if termination is None:
        termination = default_termination(problem)

    if copy_termination:
        termination = copy.deepcopy(termination)
    algorithm.termination = termination

    # actually execute the algorithm
    res = algorithm.solve()

    # store the deep copied algorithm in the result object
    res.algorithm = algorithm

    return res


def default_termination(problem):
    if problem.n_obj > 1:
        termination = MultiObjectiveDefaultTermination()
    else:
        termination = SingleObjectiveDefaultTermination()

    return termination
