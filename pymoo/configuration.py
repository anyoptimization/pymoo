import os


class Configuration:
    pass


# returns the directory to be used for imports
def get_pymoo():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
