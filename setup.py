import os
from setuptools import setup


setup(
    name="pymoo",
    version="0.0.1",
    author="Julian Blank",
    description=("Multi-Objective Optimization in Python"),
    license='MIT',
    keywords="moo,nsga",
    packages=['pymoo'],
    install_requires=['numpy']
)
