import setuptools
from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


__name__ = "pymoo"
__author__ = "Julian Blank"
__version__ = '0.2.3'
__url__ = "https://github.com/msu-coinlab/pymoo"

kwargs = dict(
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
    packages=setuptools.find_packages(exclude=['tests', 'docs', 'experiments']),
    install_requires=['pymop==0.2.3', 'numpy', 'scipy', 'matplotlib'],
    include_package_data=True,
    platforms='any'
)


def run_setup(binary=False):
    if binary:
        from Cython.Build import cythonize
        import numpy

        setup(
            include_dirs=[numpy.get_include()],
            ext_modules=cythonize("pymoo/cython/*.pyx"),
            language="c++",
            **kwargs
        ),

    else:
        setup(**kwargs)


compile_exception = None
plain_exception = None

try:
    run_setup(True)

except Exception as _compile_exception:
    compile_exception = _compile_exception

    try:
        run_setup(False)
    except Exception as _plain_exception:
        plain_exception = _plain_exception

if compile_exception is None:
    print('*' * 75)
    print("Python installation with compiled extensions succeeded.")
    print('*' * 75)
else:
    print('*' * 75)
    print("WARNING", compile_exception)
    print("WARNING: The C extension could not be compiled, speedups are not enabled.")
    print("WARNING: pip install cython numpy")
    print("WARNING: Also, make sure you have a compiler for C (gcc, clang, mscpp, ..)")
    print('*' * 75)

    if plain_exception is None:
        print("Plain Python installation succeeded.")
        print('*' * 75)
    else:
        print("WARNING", plain_exception)
        print("WARNING", "Error while installation.")
        print('*' * 75)
