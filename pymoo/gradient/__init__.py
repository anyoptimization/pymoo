import sys
import importlib

GRADIENT_MODULE = "pymoo.gradient.toolbox"

TOOLBOX = "numpy"


def activate(name):
    global TOOLBOX
    TOOLBOX = name

    if GRADIENT_MODULE in sys.modules:
        del sys.modules[GRADIENT_MODULE]
    importlib.import_module(GRADIENT_MODULE)


def deactivate():
    activate("numpy")


