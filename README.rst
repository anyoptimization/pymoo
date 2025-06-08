

.. |python| image:: https://img.shields.io/badge/python-3.10-blue.svg
   :alt: python 3.10

.. |license| image:: https://img.shields.io/badge/license-apache-orange.svg
   :alt: license apache
   :target: https://www.apache.org/licenses/LICENSE-2.0


.. |logo| image:: https://github.com/anyoptimization/pymoo-data/blob/main/logo.png?raw=true
  :target: https://pymoo.org
  :alt: pymoo


.. |animation| image:: https://github.com/anyoptimization/pymoo-data/blob/main/animation.gif?raw=true
  :target: https://pymoo.org
  :alt: pymoo


.. _Github: https://github.com/anyoptimization/pymoo
.. _Documentation: https://www.pymoo.org/
.. _Paper: https://ieeexplore.ieee.org/document/9078759




|python| |license|


|logo|



Documentation_ / Paper_ / Installation_ / Usage_ / Citation_ / Contact_



pymoo: Multi-objective Optimization in Python
====================================================================

Our open-source framework pymoo offers state of the art single- and multi-objective algorithms and many more features
related to multi-objective optimization such as visualization and decision making.


.. _Installation:

Installation
********************************************************************************

First, make sure you have a Python 3 environment installed. We recommend miniconda3 or anaconda3.

The official release is always available at PyPi:

.. code:: bash

    pip install -U pymoo


For the current developer version:

.. code:: bash

    git clone https://github.com/anyoptimization/pymoo
    cd pymoo
    pip install .


Since for speedup, some of the modules are also available compiled, you can double-check
if the compilation worked. When executing the command, be sure not already being in the local pymoo
directory because otherwise not the in site-packages installed version will be used.

.. code:: bash

    python -c "from pymoo.functions import is_compiled;print('Compiled Extensions: ', is_compiled())"


.. _Usage:

Usage
********************************************************************************

We refer here to our documentation for all the details.
However, for instance, executing NSGA2:

.. code:: python


    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.problems import get_problem
    from pymoo.optimize import minimize
    from pymoo.visualization.scatter import Scatter

    problem = get_problem("zdt1")

    algorithm = NSGA2(pop_size=100)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 200),
                   seed=1,
                   verbose=True)

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, color="red")
    plot.show()



A representative run of NSGA2 looks as follows:

|animation|



.. _Citation:

Citation
********************************************************************************

If you have used our framework for research purposes, you can cite our publication by:

| `J. Blank and K. Deb, pymoo: Multi-Objective Optimization in Python, in IEEE Access, vol. 8, pp. 89497-89509, 2020, doi: 10.1109/ACCESS.2020.2990567 <https://ieeexplore.ieee.org/document/9078759>`_
|
| BibTex:

::

    @ARTICLE{pymoo,
        author={J. {Blank} and K. {Deb}},
        journal={IEEE Access},
        title={pymoo: Multi-Objective Optimization in Python},
        year={2020},
        volume={8},
        number={},
        pages={89497-89509},
    }

.. _Contact:

Contact
********************************************************************************

Feel free to contact me if you have any questions:

| `Julian Blank <http://julianblank.com>`_  (blankjul [at] msu.edu)
| Michigan State University
| Computational Optimization and Innovation Laboratory (COIN)
| East Lansing, MI 48824, USA



