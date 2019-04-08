

Genetic Algorithm
----------------------------------
A simple genetic algorithm to solve single-objective problems.

.. literalinclude:: ../../pymoo/usage/ga.py
   :language: python


NSGA2
----------------------------------

The algorithm is coded corresponding to :cite:`Deb:2002:FEM:2221359.2221582`.


.. literalinclude:: ../../pymoo/usage/nsga2.py
   :language: python


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


.. literalinclude:: ../../pymoo/usage/nsga3.py
   :language: python

UNSGA3
----------------------------------



.. literalinclude:: ../../pymoo/usage/unsga3.py
   :language: python


RNSGA3
----------------------------------


.. literalinclude:: ../../pymoo/usage/rnsga3.py
   :language: python





Differential Evolution
----------------------------------

:cite:`Price:2005:DEP:1121631` The classical single-objective
differential evolution algorithm where different crossover variations
and methods can be defined. It is known for its good results for
effective global optimization.

.. literalinclude:: ../../pymoo/usage/de.py
   :language: python

