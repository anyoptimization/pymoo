# Function package for pymoo
# Contains both compiled (Cython) and standard (Python) implementations

import importlib

from pymoo.config import Config


def get_functions():
    from pymoo.functions.standard.non_dominated_sorting import (
        fast_non_dominated_sort,
        efficient_non_dominated_sort,
        tree_based_non_dominated_sort,
        dominance_degree_non_dominated_sort,
        find_non_dominated,
        fast_best_order_sort,
    )
    from pymoo.functions.standard.decomposition import calc_distance_to_weights
    from pymoo.functions.standard.calc_perpendicular_distance import calc_perpendicular_distance
    from pymoo.functions.standard.hv import hv
    from pymoo.functions.standard.stochastic_ranking import stochastic_ranking
    from pymoo.functions.standard.mnn import calc_mnn, calc_2nn
    from pymoo.functions.standard.pruning_cd import calc_pcd

    FUNCTIONS = {
        "fast_non_dominated_sort": {
            "python": fast_non_dominated_sort,
            "cython": "pymoo.functions.compiled.non_dominated_sorting",
        },
        "find_non_dominated": {
            "python": find_non_dominated,
            "cython": "pymoo.functions.compiled.non_dominated_sorting",
        },
        "efficient_non_dominated_sort": {
            "python": efficient_non_dominated_sort,
            "cython": "pymoo.functions.compiled.non_dominated_sorting",
        },
        "fast_best_order_sort": {
            "python": fast_best_order_sort,
            "cython": "pymoo.functions.compiled.non_dominated_sorting",
        },
        "tree_based_non_dominated_sort": {
            "python": tree_based_non_dominated_sort,
            "cython": "pymoo.functions.compiled.non_dominated_sorting",
        },
        "dominance_degree_non_dominated_sort": {
            "python": dominance_degree_non_dominated_sort,
            "cython": "pymoo.functions.compiled.non_dominated_sorting",
        },
        "calc_distance_to_weights": {
            "python": calc_distance_to_weights,
            "cython": "pymoo.functions.compiled.decomposition",
        },
        "calc_perpendicular_distance": {
            "python": calc_perpendicular_distance,
            "cython": "pymoo.functions.compiled.calc_perpendicular_distance",
        },
        "stochastic_ranking": {
            "python": stochastic_ranking,
            "cython": "pymoo.functions.compiled.stochastic_ranking",
        },
        "calc_mnn": {"python": calc_mnn, "cython": "pymoo.functions.compiled.mnn"},
        "calc_2nn": {"python": calc_2nn, "cython": "pymoo.functions.compiled.mnn"},
        "calc_pcd": {"python": calc_pcd, "cython": "pymoo.functions.compiled.pruning_cd"},
    }

    return FUNCTIONS


class FunctionLoader:
    # -------------------------------------------------
    # Singleton Pattern
    # -------------------------------------------------
    __instance = None

    @staticmethod
    def get_instance():
        if FunctionLoader.__instance is None:
            FunctionLoader.__instance = FunctionLoader()
        return FunctionLoader.__instance

    # -------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        self.is_compiled = is_compiled()
        self.mode = "auto"

        if Config.warnings["not_compiled"] and not self.is_compiled:
            print("\nCompiled modules for significant speedup can not be used!")
            print("https://pymoo.org/installation.html#installation")
            print()
            print("To disable this warning:")
            print("from pymoo.config import Config")
            print("Config.warnings['not_compiled'] = False\n")

    def load(self, func_name=None, mode=None):
        if mode is None:
            mode = self.mode

        FUNCTIONS = get_functions()

        if mode == "auto":
            mode = "cython" if self.is_compiled else "python"

        if func_name not in FUNCTIONS:
            raise Exception("Function %s not found: %s" % (func_name, FUNCTIONS.keys()))

        func = FUNCTIONS[func_name]
        if mode not in func:
            raise Exception("Module not available in %s." % mode)
        func = func[mode]

        # either provide a function or a string to the module (used for cython)
        if not callable(func):
            module = importlib.import_module(func)
            func = getattr(module, func_name)

        return func


def load_function(func_name=None, _type="auto"):
    return FunctionLoader.get_instance().load(func_name, mode=_type)


def is_compiled():
    try:
        from pymoo.functions.compiled.info import info

        if info() == "yes":
            return True
        else:
            return False
    except:
        return False
