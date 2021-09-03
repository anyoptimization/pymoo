import os

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

TESTS = os.path.join(ROOT, "tests")

RESOURCES = os.path.join(TESTS, "resources")


def path_to_test_resource(*args):
    return os.path.join(RESOURCES, *args)

