
import numpy
import setuptools
from setuptools import setup, Extension


def readme():
    with open('README.rst') as f:
        return f.read()


__name__ = "pymoo"
__author__ = "Julian Blank"
__version__ = '0.2.2'
__url__ = "https://github.com/msu-coinlab/pymoo"


def run_setup(binary=False):
    if binary:

        from Cython.Build import cythonize
        kwargs = dict(
            setup_requires=['cython'],
            include_dirs=[numpy.get_include()],
            ext_modules=cythonize(
                "pymoo/cython/*.pyx",
                language="c++"
            ),
        )
    else:
        kwargs = {}

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
        install_requires=['pymop==0.2.3', 'numpy', 'scipy', 'matplotlib'],
        include_package_data=True,
        platforms='any',
        **kwargs
    )


try:
    run_setup(True)
    print("Python installation with compiled extensions succeeded.")
except BaseException as e:
    print('*' * 75)
    print("WARNING", e)
    print("WARNING: The C extension could not be compiled, speedups are not enabled.")
    print('*' * 75)

    run_setup(False)

    print('*' * 75)
    print("Python installation succeeded.")
    print('*' * 75)
