import setuptools

from setup_ext import readme

from setup_ext import run_setup

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

run_setup(kwargs, try_to_compile_first=True)




