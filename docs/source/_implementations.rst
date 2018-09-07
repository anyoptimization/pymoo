
Algorithms
----------

**Genetic Algorithm**: A simple genetic algorithm to solve single-objective problems.

**NSGA-II** :cite:`Deb:2002:FEM:2221359.2221582`: Non-dominated sorting genetic algorithm for
bi-objective problems. The mating selection is done using the binary
tournament by comparing the rank and the crowding distance. The crowding
distance is a niching measure in a two-dimensional space which sums up
the difference to the neighbours in each dimension. The non-dominated
sorting considers the rank determined by being in the ith front and the
crowding distance to achieve a good diversity when converging.

**NSGA-III** :cite:`nsgaIII` :cite:`nsgaIII_part2`: A referenced-based algorithm used to solve
many-objective problems. The survival selection uses the perpendicular
distance to the reference directions. As normalization the boundary
intersection method is used [5].

**MOEAD/D** :cite:`Zhang07amulti-objective`: The classical MOEAD\D implementation using the
Tchebichew decomposition function.

**Differential Evolution** :cite:`Price:2005:DEP:1121631`: The classical single-objective
differential evolution algorithm where different crossover variations
and methods can be defined. It is known for its good results for
effective global optimization.

Methods
-------

**Simulated Binary Crossover** :cite:`Deb:2007:SSB:1276958.1277190`: This crossover simulates a
single-point crossover in a binary representation by using an
exponential distribution for real values. The polynomial mutation is
defined accordingly which performs basically a binary bitflip for real
numbers.