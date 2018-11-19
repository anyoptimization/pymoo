import setuptools
from setuptools import Extension

from setup_ext import readme, run_setup


def get_extension_modules():
    ext_modules = []
    for f in ["info", "non_dominated_sorting_cython", "decomposition_cython", "calc_perpendicular_distance_cython"]:
        ext_modules.append(Extension('pymoo.cython.%s' % f, sources=['pymoo/cython/%s.pyx' % f], language="c++"))
    return ext_modules


__name__ = "pymoo"
__author__ = "Julian Blank"
__version__ = '0.2.4'
__url__ = "https://github.com/msu-coinlab/pymoo"

kwargs = dict(
    name=__name__,
    version=__version__,
    author=__author__,
    url=__url__,
    python_requires='>3.3',
    author_email="blankjul@egr.msu.edu",
    description="Multi-Objective Optimization Algorithms",
    long_description=readme(),
    license='Apache License 2.0',
    keywords="optimization",
    packages=setuptools.find_packages(exclude=['tests', 'docs', 'experiments']),
    install_requires=['pymop==0.2.3', 'numpy>=1.15', 'scipy>=1.1', 'matplotlib>=3'],
    include_package_data=True,
    platforms='any'
)

run_setup(kwargs)
