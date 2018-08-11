pymoo - Multi-Objective Optimization
==================================

Installation
==================================

The test problems are uploaded to the PyPi Repository.

.. code:: bash

    pip install pymoo

For the current development version:

.. code:: bash

    git clone https://github.com/msu-coinlab/pymoo
    cd pymoo
    python setup.py install

Implementations
==================================

Algorithms
----------

**Genetic Algorithm**: A simple genetic algorithm to solve
single-objective problems.

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
==================================
.. code:: python

    import time

    import numpy as np

    from pymoo.util.plotting import plot, animate

    if __name__ == '__main__':

        # load the problem instance
        from pymop.problems.zdt import ZDT1
        problem = ZDT1(n_var=30)

        # create the algorithm instance by specifying the intended parameters
        from pymoo.algorithms.nsga2 import NSGA2
        algorithm = NSGA2(pop_size=100)

        start_time = time.time()

        # number of generations to run it
        n_gen = 200

        # solve the problem and return the results
        X, F, G = algorithm.solve(problem,
                                  evaluator=(100 * n_gen),
                                  seed=2,
                                  return_only_feasible=False,
                                  return_only_non_dominated=False,
                                  disp=True,
                                  save_history=True)

        print("--- %s seconds ---" % (time.time() - start_time))

        scatter_plot = True
        save_animation = True

        if scatter_plot:
            plot(F)

        if save_animation:
            H = np.concatenate([e['pop'].F[None, :, :] for e in algorithm.history], axis=0)
            animate('%s.mp4' % problem.name(), H, problem)

Contact
==================================
Feel free to contact me if you have any question:

| Julian Blank (blankjul [at] egr.msu.edu)
| Michigan State University
| Computational Optimization and Innovation Laboratory (COIN)
| East Lansing, MI 48824, USA
