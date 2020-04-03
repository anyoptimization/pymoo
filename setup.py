import copy
import distutils
import os
import sys
import traceback

import setuptools
from setuptools import setup, Extension
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
    python_requires='>=3.6',
    author_email="blankjul@egr.msu.edu",
    description="Multi-Objective Optimization in Python",
    license='Apache License 2.0',
    keywords="optimization",
    install_requires=['numpy>=1.15', 'scipy>=1.1', 'matplotlib>=3', 'autograd>=1.3', 'cma==2.7'],
    platforms='any',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
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


def packages():
    return ["pymoo"] + ["pymoo." + e for e in setuptools.find_packages(where='pymoo')]


data['long_description'] = readme()
data['long_description_content_type'] = 'text/x-rst'
data['packages'] = packages()


# ---------------------------------------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------------------------------------


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


def run_setup(setup_args):
    # try to add compilation to the setup - if fails just do default
    try:

        do_cythonize = False
        if "--cythonize" in sys.argv:
            do_cythonize = True
            sys.argv.remove("--cythonize")

        # copy the kwargs for compiling purpose - if it fails setup_args remain unchanged
        kwargs = copy.deepcopy(setup_args)
        kwargs['cmdclass'] = {}

        try:
            import numpy as np
            kwargs['include_dirs'] = [np.get_include()]
        except BaseException:
            raise CompilingFailed(
                "NumPy libraries must be installed for compiled extensions! Speedups are not enabled.")

        # return the object for building which allows installation with no compilation
        kwargs['cmdclass']['build_ext'] = construct_build_ext(build_ext)

        # all the modules must be finally added here
        kwargs['ext_modules'] = []
        cython_folder = os.path.join("pymoo", "cython")
        cython_files = os.listdir(cython_folder)

        # if the pyx files should be translated and then compiled
        if do_cythonize:
            from Cython.Build import cythonize
            kwargs['ext_modules'] = cythonize("pymoo/cython/*.pyx")

        # otherwise use the existing pyx files - normal case during pip installation
        else:

            # find all cpp files in czthon folder
            cpp_files = [f for f in cython_files if f.endswith(".cpp")]

            # add for each file an extension object to be compiled
            for source in cpp_files:
                ext = Extension("pymoo.cython.%s" % source[:-4], [os.path.join(cython_folder, source)])
                kwargs['ext_modules'].append(ext)

        if len(kwargs['ext_modules']) == 0:
            print('*' * 75)
            print("WARNING: No modules for compilation available. To compile pyx files, execute:")
            print("make compile-with-cython")
            print('*' * 75)
            raise CompilingFailed()

        setup(**kwargs)
        print('*' * 75)
        print("Compilation Successful.")
        print("Installation with Compilation succeeded.")
        print('*' * 75)

    except BaseException:

        # retrieve the original input arguments and execute default setup
        kwargs = setup_args
        setup(**kwargs)

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


run_setup(data)
