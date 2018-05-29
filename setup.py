from setuptools import setup

__author__ = "Julian Blank"
__version__ = '0.1.1'

try:
    import pypandoc

    long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    long_description = ''

setup(
    name="pymoo",
    version=__version__,
    author=__author__,
    author_email="blankjul@egr.msu.edu",
    description="Multi-Objective Optimization Algorithms",
    long_description=long_description,
    url="https://github.com/julesy89/pymoo",
    license='MIT',
    keywords="optimization",
    packages=['pymoo'],
    install_requires=['pymop' 'numpy', 'scipy']
)
