import importlib


class FunctionLoader:
    __instance = None

    def __init__(self) -> None:
        super().__init__()

        self.cythonize = False

        try:
            from pymoo.cython.extensions import info
            if info() == "yes":
                self.cythonize = True
            else:
                raise Exception()
        except:
            print("\nCython can not be used. Compile for speedup!\n")

    @staticmethod
    def get_instance():
        if FunctionLoader.__instance is None:
            FunctionLoader.__instance = FunctionLoader()
        return FunctionLoader.__instance

    def load(self, module, func_name=None):

        if self.cythonize:
            vals = importlib.import_module('pymoo.cython.%s_cython' % module)
        else:
            vals = importlib.import_module('pymoo.cython.%s' % module)

        if func_name is None:
            func_name = module
        func = getattr(vals, func_name)
        return func


def load_function(module, func_name=None):
    return FunctionLoader.get_instance().load(module, func_name=func_name)