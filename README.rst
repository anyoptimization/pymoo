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

    pip install Cython>=0.29 numpy>=1.15 pymoo


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


    import numpy as np

    from pymoo.optimize import minimize
    from pymoo.util import plotting
    from pymop.factory import get_problem

    # create the optimization problem
    problem = get_problem("zdt1")
    pf = problem.pareto_front()

    res = minimize(problem,
                   method='nsga2',
                   method_args={'pop_size': 100},
                   termination=('n_gen', 200),
                   pf=pf,
                   save_history=True,
                   disp=True)

    plot = True
    if plot:
        plotting.plot(pf, res.F, labels=["Pareto-front", "F"])

    # set true if you want to save a video
    animate = False
    if animate:
        from pymoo.util.plotting import animate as func_animtate
        H = np.concatenate([e.pop.get("F")[None, :] for e in res.history], axis=0)
        func_animtate('%s.mp4' % problem.name(), H, problem)



Contact
====================================================================
Feel free to contact me if you have any question:

| Julian Blank (blankjul [at] egr.msu.edu)
| Michigan State University
| Computational Optimization and Innovation Laboratory (COIN)
| East Lansing, MI 48824, USA

