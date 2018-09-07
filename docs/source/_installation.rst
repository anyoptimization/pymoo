
Before using the installer check if the following requirements are fulfilled:

Python Version 3
.. code:: bash
    python --version

pip>=9.0.0
.. code:: bash
    pip --version

cython:
.. code:: bash
    pip install cython



The test problems are uploaded to the PyPi Repository.

.. code:: bash

    pip install pymoo

For the current development version:

.. code:: bash

    git clone https://github.com/msu-coinlab/pymoo
    cd pymoo
    python setup.py install


Just locally to be used directly in another project:

.. code:: bash

    git clone https://github.com/msu-coinlab/pymoo
    cd pymoo
    pyhton setup.py build_ext --inplace