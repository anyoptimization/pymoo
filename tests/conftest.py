import importlib
import sys


def pytest_addoption(parser):
    parser.addoption("--overwrite", action="store_true")


# sys.path = [e for e in sys.path if "pymoo" not in e]
# print(sys.path)
# print(importlib.util.find_spec("pymoo.cython.non_dominated_sorting"))
