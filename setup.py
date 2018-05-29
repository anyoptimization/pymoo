from setuptools import find_packages
from setuptools import setup

__version__ = '0.1.1'


setup(
    name="pymoo",
    version=__version__,
    author="Julian Blank",
    description=("Multi-Objective Optimization"),
    license='MIT',
    keywords="moo,nsga",
    packages=find_packages(),
    install_requires=['numpy', 'pybind11>=2.2'],
)
