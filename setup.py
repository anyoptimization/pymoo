import setuptools
import setuptools.command.build_ext
import Cython.Build

import numpy

setuptools.setup(
    ext_modules=Cython.Build.cythonize("src/pymoo/cython/*.pyx"),
    include_dirs=[numpy.get_include()],
)
