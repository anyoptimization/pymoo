pymoo - Multi-Objective Optimization Framework
====================================================================

You can find the detailed documentation here:
https://www.egr.msu.edu/coinlab/blankjul/pymoo/


Installation
====================================================================

First, make sure you have a python environment installed. We recommend miniconda3 or anaconda3.

.. code:: bash

    conda --version

Then from scratch create a virtual environment for pymoo:

.. code:: bash

    conda create -n pymoo -y python==3.7.1 cython numpy
    conda activate pymoo


For the current stable release please execute:

.. code:: bash

    pip install pymoo

For the current development version:

.. code:: bash

    git clone https://github.com/msu-coinlab/pymoo
    cd pymoo
    pip install .

Since for speedup some of the modules are also available compiled you can double check
if the compilation worked:

.. code:: bash

    python -c 'from pymoo.cython.function_loader import is_compiled;print("Compiled Extentions: ", is_compiled())'




Usage
==================================

We refer here to our documentation for all the details.
However, for instance executing NSGA2:

.. code:: python

    
    from pymoo.optimize import minimize
    from pymoo.util import plotting
    from pymop.factory import get_problem

    # create the optimization problem
    problem = get_problem("zdt1")

    # solve the given problem using an optimization algorithm (here: nsga2)
    res = minimize(problem,
                   method='nsga2',
                   method_args={'pop_size': 100},
                   termination=('n_gen', 200),
                   pf=problem.pareto_front(100),
                   save_history=False,
                   disp=True)
    plotting.plot(res.F)



Contact
====================================================================
Feel free to contact me if you have any question:

| Julian Blank (blankjul [at] egr.msu.edu)
| Michigan State University
| Computational Optimization and Innovation Laboratory (COIN)
| East Lansing, MI 48824, USA

