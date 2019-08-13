pymoo - Multi-Objective Optimization Framework
====================================================================

You can find the detailed documentation here: https://pymoo.org


|gitlab| |python| |license|


.. |gitlab| image:: https://gitlab.msu.edu/blankjul/pymoo/badges/master/pipeline.svg
   :alt: build status
   :target: https://gitlab.msu.edu/blankjul/pymoo/commits/master

.. |python| image:: https://img.shields.io/badge/python-3.6-blue.svg
   :alt: python 3.6

.. |license| image:: https://img.shields.io/badge/license-apache-orange.svg
   :alt: license apache
   :target: https://www.apache.org/licenses/LICENSE-2.0


We are currently working on a paper about *pymoo*.
Meanwhile, if you have used our framework for research purposes, please cite us with:

::

   @misc{pymoo,
       author = {Julian Blank and Kalyanmoy Deb},
       title = {pymoo - {Multi-objective Optimization in Python}},
       howpublished = {https://pymoo.org}
   }



Installation
====================================================================

First, make sure you have a Python 3 environment installed. We recommend miniconda3 or anaconda3.

The official release is always available at PyPi:

.. code:: bash

    pip install -U pymoo


For the current developer version:

.. code:: bash

    git clone https://github.com/msu-coinlab/pymoo
    cd pymoo
    pip install .


Since for speedup some of the modules are also available compiled you can double check
if the compilation worked. When executing the command be sure not already being in the local pymoo
directory because otherwise not the in site-packages installed version will be used.

.. code:: bash

    python -c "from pymoo.cython.function_loader import is_compiled;print('Compiled Extensions: ', is_compiled())"


Usage
==================================

We refer here to our documentation for all the details.
However, for instance executing NSGA2:

.. code:: python

    
    from pymoo.algorithms.nsga2 import NSGA2
    from pymoo.factory import get_problem
    from pymoo.optimize import minimize
    from pymoo.visualization.scatter import Scatter

    problem = get_problem("zdt3")
    pf = problem.pareto_front(n_points=200, flatten=False, use_cache=False)

    algorithm = NSGA2(pop_size=100, elimate_duplicates=True)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 200),
                   seed=1,
                   verbose=False)

    plot = Scatter()
    plot.add(pf, plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, color="red")
    plot.show()



Contact
====================================================================

Feel free to contact me if you have any question:

| Julian Blank (blankjul [at] egr.msu.edu)
| Michigan State University
| Computational Optimization and Innovation Laboratory (COIN)
| East Lansing, MI 48824, USA
