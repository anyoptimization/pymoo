"""Convenience function for minimization using optimization algorithms."""

import copy


def minimize(
    problem,
    algorithm,
    termination=None,
    copy_algorithm=True,
    copy_termination=True,
    **kwargs,
):
    """Minimization of function of one or more variables, objectives and constraints.

    This is used as a convenience function to execute several algorithms with default
    settings which turned out to work for a test case. However, evolutionary computations
    utilize the idea of customizing a meta-algorithm. Customizing the algorithm using
    the object-oriented interface is recommended to improve convergence.

    Args:
        problem: A problem object which is defined using pymoo.
        algorithm: The algorithm object that should be used for the optimization.
        termination: The termination criterion that is used to stop the algorithm.
        copy_algorithm: Whether the algorithm object should be copied before optimization.
        copy_termination: Whether the termination object should be copied.
        **kwargs: Additional arguments passed to algorithm.setup(), such as seed,
            verbose, display, callback, save_history.

    Returns:
        The optimization result represented as an object.
    """
    # create a copy of the algorithm object to ensure no side effects
    if copy_algorithm:
        algorithm = copy.deepcopy(algorithm)

    # initialize the algorithm object given a problem - if not set already
    if algorithm.problem is None:
        if termination is not None:
            if copy_termination:
                termination = copy.deepcopy(termination)

            kwargs["termination"] = termination

        algorithm.setup(problem, **kwargs)

    # actually execute the algorithm
    res = algorithm.run()

    # store the deep copied algorithm in the result object
    res.algorithm = algorithm

    return res
