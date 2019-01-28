import importlib


class FunctionLoader:
    __instance = None

    def is_compiled(self):
        try:
            from pymoo.cython.info import info
            if info() == "yes":
                return True
            else:
                return False
        except:
            return False

    def __init__(self) -> None:
        super().__init__()
        self.cythonize = self.is_compiled()

        if not self.cythonize:
            print("\nCompiled libraries can not be used. Compile for speedup!\n")

    @staticmethod
    def get_instance():
        if FunctionLoader.__instance is None:
            FunctionLoader.__instance = FunctionLoader()
        return FunctionLoader.__instance

    def load(self, module, func_name=None):

        if self.cythonize:

            try:
                vals = importlib.import_module('pymoo.cython.%s_cython' % module)
            except:
                print("Compiled module %s cannot be loaded." % module)
                vals = importlib.import_module('pymoo.cython.%s' % module)

        else:
            vals = importlib.import_module('pymoo.cython.%s' % module)

        if func_name is None:
            func_name = module
        func = getattr(vals, func_name)
        return func


def load_function(module, func_name=None):
    return FunctionLoader.get_instance().load(module, func_name=func_name)


def is_compiled():
    return FunctionLoader.get_instance().is_compiled()
