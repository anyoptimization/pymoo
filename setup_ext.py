from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError)


def readme():
    with open('README.rst') as f:
        return f.read()


def get_extension_modules():
    ext_modules = []
    for f in ["info", "non_dominated_sorting_cython", "decomposition_cython", "calc_perpendicular_distance_cython"]:
        ext_modules.append(Extension('pymoo.cython.%s' % f, sources=['pymoo/cython/%s.pyx' % f], language="c++"))
    return ext_modules


class BuildFailed(Exception):
    pass


def construct_build_ext(build_ext):
    class WrappedBuildExt(build_ext):
        # This class allows C extension building to fail.
        def run(self):
            try:
                build_ext.run(self)
            except DistutilsPlatformError as x:
                raise BuildFailed(x)

        def build_extension(self, ext):
            try:
                build_ext.build_extension(self, ext)
            except ext_errors as x:
                raise BuildFailed(x)

    return WrappedBuildExt


def run_setup(setup_args):
    try:

        # copy the kwargs
        kwargs = dict(setup_args)
        kwargs['cmdclass'] = {}

        # import numpy and cython for compilation
        try:
            import numpy as np
            from Cython.Build import cythonize
        except:
            raise BuildFailed("The C extension could not be compiled, speedups are not enabled")

        kwargs['cmdclass']['build_ext'] = construct_build_ext(build_ext)
        kwargs['ext_modules'] = cythonize("pymoo/cython/*.pyx")
        kwargs['include_dirs'] = [np.get_include()]

        setup(**kwargs)

        print('*' * 75)
        print("Compilation Successful.")
        print("Compiled pymoo installation succeeded.")
        print('*' * 75)

    except BuildFailed as ex:

        setup(**setup_args)

        print('*' * 75)
        print("WARNING:", ex)
        print("WARNING: For the compiled libraries cython and numpy is required. Please make sure they are installed")
        print("WARNING: pip install cython numpy")
        print("WARNING: Also, make sure you have a compiler for C++!")
        print('*' * 75)
        print("Plain Python installation succeeded.")
        print('*' * 75)
