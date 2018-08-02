
def minimize(fun, xl, xu, method='auto', fun_args={}, method_args={},
             callback=None, display=False):
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

    fun_args : dict
        Additional arguments if necessary to evaluate the solutions (constants, random seed, ...)

    xl : numpy.array
        The lower boundaries of variables as a 1d array

    xu : numpy.array
        The upper boundaries of variables as a 1d array

    callback : callable, optional
        Called after each iteration, as ``callback(D)``, where ``D`` is a dictionary with
        algorithm information, such as the current design, objective and constraint space.

    display : bool
        Whether to display each generation the current result or not.

    Returns
    -------
    res : Result
        The optimization result represented as a ``Result`` object.


    """

    pass
