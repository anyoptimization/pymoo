import os
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="pymoo",
    version="0.0.1",
    author="Yashes Dhebar, Julian Blank",
    description=("Multi-Objective Optimization in Python"),
    license='MIT',
    keywords="moo,nsga",
    packages=['pymoo', 'tests'],
    long_description=read('README'),
    install_requires=['numpy', 'pygmo']
)
