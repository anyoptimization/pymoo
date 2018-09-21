import numpy
import setuptools
from Cython.Build import cythonize
from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

__name__ = "pymoo"
__author__ = "Julian Blank"
__version__ = '0.2.1'
__url__ = "https://github.com/msu-coinlab/pymoo"



setup(
    name=__name__,
    version=__version__,
    author=__author__,
    url=__url__,
    python_requires='>3.3.0',
    author_email="blankjul@egr.msu.edu",
    description="Multi-Objective Optimization Algorithms",
    long_description=readme(),
    license='Apache License 2.0',
    keywords="optimization",
    packages=setuptools.find_packages(exclude=['tests', 'docs']),
    setup_requires=['cython'],
    install_requires=['pymop==0.2.1', 'numpy', 'scipy', 'matplotlib'],
    ext_modules=cythonize(
        "pymoo/cython/*.pyx",
        language="c++"
    ),
    include_dirs=[numpy.get_include()],
    include_package_data=True,
    platforms='any',
)
