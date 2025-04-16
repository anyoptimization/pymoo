import numpy
import setuptools
import Cython.Build

setuptools.setup(
    ext_modules=Cython.Build.cythonize("pymoo/cython/*.pyx"),
    include_dirs=[numpy.get_include()],
)
