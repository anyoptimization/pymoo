from setup_ext import readme, run_setup


__name__ = "pymoo"
__author__ = "Julian Blank"
__version__ = '0.3.1.dev'
__url__ = "https://pymoo.org"

kwargs = dict(
    name=__name__,
    version=__version__,
    author=__author__,
    url=__url__,
    python_requires='>3.5',
    author_email="blankjul@egr.msu.edu",
    description="Multi-Objective Optimization in Python",
    long_description=readme(),
    license='Apache License 2.0',
    keywords="optimization",
    install_requires=['numpy>=1.15', 'scipy>=1.1', 'matplotlib>=3', 'autograd>=1.3'],
    include_package_data=True,
    platforms='any',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics'
    ]
)

run_setup(kwargs)
