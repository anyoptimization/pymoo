---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _installation:
```

.. meta::
   :description: How to install pymoo, an open-source multi-objective optimization framework in Python.

+++

.. meta::
   :keywords: pymoo, PyPI, Python, Framework, Multi-objective Optimization

+++

# Installation

+++

If you have not really worked with Python before, we recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://www.anaconda.com). Please follow the installation guides to set up a Python environment. For convenience, we also provide a quick guide [below](#Conda).

+++

## Stable

+++

To install the most recent stable release of *pymoo* please use **PyPI**

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. code:: bash

    pip install -U pymoo
```

For MacOSX and Windows **compiled** packages are available.  For Linux the package will be compiled during installation (make sure that **NumPy** is installed before executing the **pip** command). If you encounter any difficulties during compilation or you prefer to compile the package by yourself please see our guide [below](#Compilation).

+++

## Optional Dependencies

+++

*pymoo* provides optional dependency groups for additional functionality:

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. code:: bash

    # For visualization features (matplotlib-based plotting)
    pip install -U pymoo[visualization]
    
    # For parallelization features (joblib, dask, ray)
    pip install -U pymoo[parallelization]
    
    # For all optional features
    pip install -U pymoo[full]
```

**Visualization**: Includes matplotlib for creating plots, animations, and interactive visualizations of optimization results.

**Parallelization**: Includes joblib, dask, and ray for distributed and parallel evaluation of objective functions across multiple cores or machines.

+++

To quickly check if the compilation was successful you can use the following command:

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. code:: bash

    python -c "from pymoo.functions import is_compiled;print('Compiled Extensions: ', is_compiled())"
```

## Release Candidate

+++

To install the current release candidate you simply have to add `--pre` to the installation command:

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. code:: bash

    pip install --pre -U pymoo
```

## Development

+++

If you intend to use not the latest development, select the branch you intend to use (below it is master) and install it by:

```{raw-cell}
:raw_mimetype: text/restructuredtext


.. code:: bash

    pip install numpy
    git clone https://github.com/anyoptimization/pymoo
    cd pymoo
    make compile
    pip install .
```

## Compilation

+++

The *pymoo* framework can be run with and without compiled modules. Some computationally more
expensive functions have been implemented using [Cython](https://github.com/cython/cython) for speedup. 

The compilation requires *NumPy* to be installed because its header files are needed. 
You can use the make command below directly:

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. code:: bash

    make compile
```

Or execute the same command by

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. code:: bash

    python setup.py build_ext --inplace
```

Or if Cython shall be used to create the `cpp` files from scratch use

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. code:: bash

    python setup.py build_ext --inplace --cythonize
```

All the commands above will show you detailed error messages if the compilation has failed.

+++

**Windows**: For Windows you have to install C++ Distributable Library to compile the modules available [here](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads).

+++

## Conda

+++

Please check if conda is available in the command line:

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. code:: bash

    conda --version
```

If you are already familiar with conda you might want to use an environment you have
already created, or you might need *pymoo* in an existing one.

Otherwise, create a new python environment with NumPy preinstalled and activate it:

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. code:: bash

    conda create -n pymoo -y python==3.9 numpy
    conda activate pymoo
    pip install -U pymoo
```

If the environment is not set up correctly, the installation will NOT fail and you
will still be able to use *pymoo* without the benefit of the compiled modules.
