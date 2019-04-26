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
__version__ = '0.3.0'
__url__ = "https://github.com/msu-coinlab/pymoo"

kwargs = dict(
    name=__name__,
    version=__version__,
    author=__author__,
    url=__url__,
    python_requires='>3.3',
    author_email="blankjul@egr.msu.edu",
    description="Multi-Objective Optimization in Python",
    long_description=readme(),
    license='Apache License 2.0',
    keywords="optimization",
    packages=setuptools.find_packages(exclude=['tests', 'doc', 'experiments']),
    install_requires=['pymop==0.2.4', 'numpy>=1.15', 'scipy>=1.1', 'matplotlib>=3'],
    include_package_data=True,
    platforms='any',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics'
    ]
)

run_setup(kwargs)
