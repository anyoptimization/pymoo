import argparse
import copy
import sysconfig
import os
import sys
import traceback

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


# ---------------------------------------------------------------------------------------------------------
# GENERAL
# ---------------------------------------------------------------------------------------------------------


data = dict(
    packages=find_packages(include=['pymoo', 'pymoo.*']),
    include_package_data=True,
    exclude_package_data={
        '': ['*.pyx'],
    },
    platforms='any',
)


# ---------------------------------------------------------------------------------------------------------
# OPTIONS
# ---------------------------------------------------------------------------------------------------------


parser = argparse.ArgumentParser()
args, _ = parser.parse_known_args()

sys.argv = [e for e in sys.argv if not e.lstrip("-") in args]



# ============================================================
# Module for Compilation - Throws an Exception if Failing
# ============================================================


# exception that is thrown when the build fails
class CompilingFailed(Exception):
    pass


# try to compile, if not possible throw exception
def construct_build_ext(build_ext):
    class WrappedBuildExt(build_ext):
        def run(self):
            try:
                build_ext.run(self)
            except BaseException as e:
                raise CompilingFailed(e)

        def build_extension(self, ext):
            try:
                build_ext.build_extension(self, ext)
            except BaseException as e:
                raise CompilingFailed(e)

    return WrappedBuildExt


# ============================================================
# SETUP
# ============================================================


from Cython.Build import cythonize

ext = cythonize("pymoo/cython/*.pyx")

data['ext_modules'] = ext

try:
    import numpy as np

    data['include_dirs'] = [np.get_include()]
except BaseException:
    raise CompilingFailed(
        "NumPy libraries must be installed for compiled extensions! Speedups are not enabled.")

# return the object for building which allows installation with no compilation
data['cmdclass'] = dict(build_ext=construct_build_ext(build_ext))

setup(**data)
print('*' * 75)
print("Compilation Successful.")
print("Installation with compiled libraries succeeded.")
print('*' * 75)

