import numpy
import setuptools
from Cython.Build import cythonize
from setuptools import setup

__author__ = "Julian Blank"
__version__ = '0.2.1-dev8'
__url__ = "https://github.com/msu-coinlab/pymoo"

try:
    import pypandoc

    long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    long_description = ''

setup(
    name="pymoo",
    version=__version__,
    author=__author__,
    python_requires='>3.3.0',
    author_email="blankjul@egr.msu.edu",
    description="Multi-Objective Optimization Algorithms",
    long_description=long_description,
    url=__url__,
    license='Apache License 2.0',
    keywords="optimization",
    packages=setuptools.find_packages(include="*.pyx", exclude=["tests", "docs"]),
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
