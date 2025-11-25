import numpy
import setuptools
import Cython.Build

setuptools.setup(
    ext_modules=Cython.Build.cythonize("pymoo/functions/compiled/*.pyx", force=True),
    include_dirs=[numpy.get_include()],
)
