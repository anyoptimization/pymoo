

GeneticAlgorithm
----------------------------------
A simple genetic algorithm to solve single-objective problems.

.. code:: python

    res = minimize(problem,
                   method='ga',
                   method_args={'pop_size': 100},
                   )


NSGA2
----------------------------------

The algorithm is coded corresponding to :cite:`Deb:2002:FEM:2221359.2221582`.


.. code:: python

    res = minimize(problem,
                   method='nsga2',
                   method_args={'pop_size': 100},
                   )


Non-dominated sorting genetic algorithm for
bi-objective problems. The mating selection is done using the binary
tournament by comparing the rank and the crowding distance. The crowding
distance is a niching measure in a two-dimensional space which sums up
the difference to the neighbours in each dimension. The non-dominated
sorting considers the rank determined by being in the ith front and the
crowding distance to achieve a good diversity when converging.


NSGA3
----------------------------------
:cite:`nsgaIII` :cite:`nsgaIII_part2` A referenced-based algorithm used to solve
many-objective problems. The survival selection uses the perpendicular
distance to the reference directions. As normalization the boundary
intersection method is used.


.. code:: python

    res = minimize(problem,
                   method='nsga3',
                   method_args={'ref_dirs': 2134124},
                   )



RNSGA3
----------------------------------


.. code:: python

    res = minimize(problem,
                   method='rnsga3',
                   method_args={'ref_points' : None, 'n_ref_points' : None},
                   )



MOEAD
----------------------------------
:cite:`Zhang07amulti-objective` The classical MOEAD\D implementation using the
Tchebichew decomposition function.




Differential Evolution
----------------------------------

:cite:`Price:2005:DEP:1121631` The classical single-objective
differential evolution algorithm where different crossover variations
and methods can be defined. It is known for its good results for
effective global optimization.