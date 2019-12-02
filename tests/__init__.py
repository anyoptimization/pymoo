import os

from pymoo.configuration import get_pymoo


def get_pymoo_test():
    return os.path.join(get_pymoo(), "tests")


def path_to_test_resources(*args):
    return os.path.join(get_pymoo_test(), "resources", *args)