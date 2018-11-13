from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


def setup_with_compilation(kwargs):
    from Cython.Build import cythonize
    import numpy

    setup(
        include_dirs=[numpy.get_include()],
        ext_modules=cythonize("pymoo/cython/*.pyx", language="c++"),
        **kwargs
    )


def run_setup(kwargs, try_to_compile_first=True):

    if not try_to_compile_first:
        setup(**kwargs)

    else:

        compile_exception = None
        plain_exception = None

        try:
            setup_with_compilation(kwargs)

        except Exception as _compile_exception:
            compile_exception = _compile_exception

            try:
                setup(**kwargs)
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
