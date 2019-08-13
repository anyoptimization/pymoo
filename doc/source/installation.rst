.. _installation:

Installation
==============================================================================

The *pymoo* framework can be run with and without compiled modules. Some computationally more
expensive function have been implemented using `Cython <https://github.com/cython/cython>`_
for speedup. To figure out what version is used after the installation please see `Plain/Compiled Modules`_ section.


Setting up the Python environment using Conda
------------------------------------------------------------------------------

Here, we are setting up the environment in order to be able to use the speedup of Cython.
Therefore, before using the install command, the environment needs to be set up.
We recommend using `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or
`anaconda <https://www.anaconda.com>`_.

Please check if conda is available in your current terminal:

.. code:: bash

    conda --version

If you are already familiar with conda you might want to use an environment you have
already created, or you might need *pymoo* in an existing one.


Otherwise, create a new python environment and activate it:

.. code:: bash

    conda create -n pymoo -y python==3.6
    conda activate pymoo


Furthermore, make sure NumPy is installed:

.. code:: bash

    pip install numpy


If the environment is not setup correct, the installation will NOT fail and you
will still be able to use *pymoo* without the benefit of the compiled modules.


PyPi
------------------------------------------------------------------------------


To use the current stable release of *pymoo* use

.. code:: bash

    pip install -U pymoo

If you have already installed an older version of the framework you can force
an update by using the *-U* option.


Development
------------------------------------------------------------------------------

If you like to use our current development version or like to contribute to
our framework you can install the current version on GitHub by

.. code:: bash

    git clone https://github.com/msu-coinlab/pymoo
    cd pymoo
    pip install .


To compile the modules or see an output log:

.. code:: bash

    pip install Cython
    make compile-with-cython

This translates the pyx files to cpp files and then compiles them. If anything fails
this will provide more details about why this has happened.
    

Plain/Compiled Modules
------------------------------------------------------------------------------

As said above, the *pymoo* installation will not fail if the modules are not
compiled successfully, but no speedup will be available. To check if the compilation
has worked during the installation, you can use the following command:

.. code:: bash

    python -c "from pymoo.util.function_loader import is_compiled;print('Compiled Extensions: ', is_compiled())"

