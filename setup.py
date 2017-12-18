from setuptools import setup, find_packages




setup(
    name="pymoo",
    version="0.0.1",
    author="Julian Blank",
    description=("Multi-Objective Optimization in Python"),
    license='MIT',
    keywords="moo,nsga",
    packages=find_packages(),
    install_requires=['numpy']
)
