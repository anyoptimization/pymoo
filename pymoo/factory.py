"""
NOTE: The factory methods have been deprecated since version 0.5.0.
The reason is the additional maintenance and the fact that the individual constructor parameters are kept hidden.

Only get_termination and get_problem are still active and are now implemented in the corresponding modules.
They will be routed through this factory method.

"""

from deprecated import deprecated


@deprecated("Please use `from pymoo.termination import get_termination`")
def get_termination(name, *args, **kwargs):
    from pymoo.termination import get_termination as func
    return func(name, *args, **kwargs)


@deprecated("Please use `from pymoo.util.ref_dirs import get_reference_directions`")
def get_reference_directions(name, *args, **kwargs):
    from pymoo.util.ref_dirs import get_reference_directions as func
    return func(name, *args, **kwargs)


@deprecated("Please use `from pymoo.problems import get_problem\n`")
def get_problem(name, *args, **kwargs):
    from pymoo.problems import get_problem as func
    return func(name, *args, **kwargs)


def get_algorithm(*args, **kwargs):
    raise Exception("The method `get_algorithm` has been deprecated since 0.6.0\n"
                    "Please use the object-oriented interface.")


def get_sampling(*args, **kwargs):
    raise Exception("The method `get_sampling` has been deprecated since 0.6.0\n"
                    "Please use the object-oriented interface:\n"
                    "https://pymoo.org/operators/selection.html")


def get_selection(*args, **kwargs):
    raise Exception("The method `get_selection` has been deprecated since 0.6.0\n"
                    "Please use the object-oriented interface:\n"
                    "https://pymoo.org/operators/selection.html")


def get_crossover(*args, **kwargs):
    raise Exception("The method `get_crossover` has been deprecated since 0.6.0\n"
                    "Please use the object-oriented interface:\n"
                    "https://pymoo.org/operators/crossover.html")


def get_mutation(*args, **kwargs):
    raise Exception("The method `get_mutation` has been deprecated since 0.6.0\n"
                    "Please use the object-oriented interface:\n"
                    "https://pymoo.org/operators/mutation.html")


def get_visualization(*args, **kwargs):
    raise Exception("The method `get_visualization` has been deprecated since 0.6.0\n"
                    "Please use the object-oriented interface:\n"
                    "https://pymoo.org/visualization/index.html")


def get_performance_indicator(*args, **kwargs):
    raise Exception("The method `get_performance_indicator` has been deprecated since 0.6.0\n"
                    "Please use the object-oriented interface:\n"
                    "https://pymoo.org/misc/indicators.html")


def get_decision_making(*args, **kwargs):
    raise Exception("The method `get_decision_making` has been deprecated since 0.6.0\n"
                    "Please use the object-oriented interface.")
