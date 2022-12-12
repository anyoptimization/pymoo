import argparse
import copy
import distutils
import os
import sys
import traceback

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

from pymoo.version import __version__

# ---------------------------------------------------------------------------------------------------------
# GENERAL
# ---------------------------------------------------------------------------------------------------------


__name__ = "pymoo"
__author__ = "Julian Blank"
__url__ = "https://pymoo.org"
data = dict(
    name=__name__,
    version=__version__,
    author=__author__,
    url=__url__,
    python_requires='>=3.7',
    author_email="blankjul@msu.edu",
    description="Multi-Objective Optimization in Python",
    license='Apache License 2.0',
    keywords="optimization",
    # packages=["pymoo"] + ["pymoo." + e for e in find_packages(where='pymoo')],
    packages=find_packages(include=['pymoo', 'pymoo.*']),
    include_package_data=True,
    exclude_package_data={
        '': ['*.c', '*.cpp', '*.pyx'],
    },
    install_requires=['numpy>=1.15',
                      'scipy>=1.1',
                      'matplotlib>=3',
                      'autograd>=1.4',
                      'cma==3.2.2',
                      'alive-progress',
                      'dill',
                      'Deprecated'],
    platforms='any',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics'
    ]
)


# ---------------------------------------------------------------------------------------------------------
# OTHER METADATA
# ---------------------------------------------------------------------------------------------------------


# update the readme.rst to be part of setup
def readme():
    with open('README.rst') as f:
        return f.read()


data['long_description'] = readme()
data['long_description_content_type'] = 'text/x-rst'

# ---------------------------------------------------------------------------------------------------------
# OPTIONS
# ---------------------------------------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('--nopyx', dest='nopyx', action='store_true',
                    help='Whether the pyx files shall be considered at all.')
parser.add_argument('--nocython', dest='nocython', action='store_true', help='Whether pyx files shall be cythonized.')
parser.add_argument('--nolibs', dest='nolibs', action='store_true', help='Whether the libraries should be compiled.')
args, _ = parser.parse_known_args()

sys.argv = [e for e in sys.argv if not e.lstrip("-") in args]


# ============================================================
# MacOSX FIX for compiling modules
# ============================================================

def is_new_osx():
    name = distutils.util.get_platform()
    if sys.platform != "darwin":
        return False
    elif name.startswith("macosx-10"):
        minor_version = int(name.split("-")[1].split(".")[1])
        if minor_version >= 7:
            return True
        else:
            return False
    else:
        return False


# fix compiling for new macosx!
if is_new_osx():
    os.environ['CFLAGS'] = '-stdlib=libc++'


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

ROOT = os.path.dirname(os.path.realpath(__file__))
backup = copy.deepcopy(data)

try:

    if args.nopyx:
        ext = []
    elif args.nocython:
        path = os.path.join(ROOT, "pymoo", "cython")
        pyx = [os.path.join(path, f) for f in os.listdir() if f.endswith(".pyx")]
        ext = [Extension(f"pymoo.cython.{source[:-4]}", [source]) for source in pyx]
    else:
        from Cython.Build import cythonize

        ext = cythonize("pymoo/cython/*.pyx")

    if not args.nolibs:

        if len(ext) > 0:
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

except BaseException:

    # retrieve the original input arguments and execute default setup
    setup(**backup)

    # get information why compiling has failed
    ex_type, ex_value, ex_traceback = sys.exc_info()

    print('*' * 75)
    print("WARNING: Compilation Failed.")
    print("WARNING:", ex_type)
    print("WARNING:", ex_value)
    print()
    print("=" * 75)
    traceback.print_exc()
    print("=" * 75)
    print()
    print("WARNING: For the compiled libraries numpy is required. Please make sure they are installed")
    print("WARNING: pip install numpy")
    print("WARNING: Also, make sure you have a compiler for C++!")

    print('*' * 75)
    print("Plain Python installation succeeded.")
    print('*' * 75)
