import typing

if typing.TYPE_CHECKING:
    from pymoo.core.termination import Termination


def get_termination(name, *args, **kwargs) -> "Termination":
    from pymoo.termination.default import DefaultMultiObjectiveTermination, DefaultSingleObjectiveTermination
    from pymoo.termination.max_eval import MaximumFunctionCallTermination
    from pymoo.termination.max_gen import MaximumGenerationTermination
    from pymoo.termination.max_time import TimeBasedTermination
    from pymoo.termination.fmin import MinimumFunctionValueTermination

    TERMINATION = {
        "n_eval": MaximumFunctionCallTermination,
        "n_evals": MaximumFunctionCallTermination,
        "n_gen": MaximumGenerationTermination,
        "n_iter": MaximumGenerationTermination,
        "fmin": MinimumFunctionValueTermination,
        "time": TimeBasedTermination,
        "soo": DefaultSingleObjectiveTermination,
        "moo": DefaultMultiObjectiveTermination,
    }

    if name not in TERMINATION:
        raise Exception("Termination not found.")

    return TERMINATION[name](*args, **kwargs)
