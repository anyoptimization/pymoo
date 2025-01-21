from os.path import dirname, realpath

from pymoo.version import __version__


class Config:
    """
    The configuration of this package in general providing the place
    for declaring global variables.
    """

    # the root directory where the package is located at
    root = dirname(realpath(__file__))

    warnings = {
        "not_compiled": True
    }

    # whether a warning should be printed if compiled modules are not available
    show_compile_hint = True

    # whether when import a file the doc should be parsed - only activate when creating doc files
    parse_custom_docs = False

    # a method defining the endpoint to load data remotely - default from GitHub repo
    @classmethod
    def data(cls):
        return f"https://raw.githubusercontent.com/anyoptimization/pymoo-data/main/"


# returns the directory to be used for imports
def get_pymoo():
    return dirname(Config.root)
