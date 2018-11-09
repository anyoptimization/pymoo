pymoo - Multi-Objective Optimization Framework
====================================================================


You can find the detailed documentation here:
https://www.egr.msu.edu/coinlab/blankjul/pymoo/


Requirements
====================================================================

Before using the installer check if the following requirements are fulfilled:

Python 3

.. code:: bash

    python --version

pip>=9.0.0

.. code:: bash

    pip --version

Cython:

.. code:: bash

    pip install cython



Installation
====================================================================

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

Implementations
====================================================================

Algorithms
----------

**Genetic Algorithm**: A simple genetic algorithm to solve single-objective problems.

**NSGA-II** : Non-dominated sorting genetic algorithm for
bi-objective problems. The mating selection is done using the binary
tournament by comparing the rank and the crowding distance. The crowding
distance is a niching measure in a two-dimensional space which sums up
the difference to the neighbours in each dimension. The non-dominated
sorting considers the rank determined by being in the ith front and the
crowding distance to achieve a good diversity when converging.

**NSGA-III** : A referenced-based algorithm used to solve
many-objective problems. The survival selection uses the perpendicular
distance to the reference directions. As normalization the boundary
intersection method is used [5].

**MOEAD/D** : The classical MOEAD\D implementation using the
Tchebichew decomposition function.

**Differential Evolution** : The classical single-objective
differential evolution algorithm where different crossover variations
and methods can be defined. It is known for its good results for
effective global optimization.

Methods
-------

**Simulated Binary Crossover** : This crossover simulates a
single-point crossover in a binary representation by using an
exponential distribution for real values. The polynomial mutation is
defined accordingly which performs basically a binary bitflip for real
numbers.

Usage
====================================================================

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

