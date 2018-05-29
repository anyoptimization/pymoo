# pymoo - Multi-Objective Optimization for Python

This framework provides multi-objective optimization algorithms for python. 
Either define your own problems to be solved or use our test problems defined in [pymop](https://github.com/julesy89/pymop).


# Installation


The test problems are uploaded to the PyPi Repository.

```bash
pip install pymoo
```



# Usage


```python

# define a problem to be solved


```



# Implementations


## Algorithms

**Genetic Algorithm**: A simple genetic algorithm to solve single-objective problems. 

**NSGA-II** [1]: Non-dominated sorting genetic algorithm for bi-objective problems. The mating selection is done using the binary tournament by comparing the rank and the crowding distance. 
The crowding distance is a niching measure in a two-dimensional space which sums up the difference to the neighbours in each dimension.
The non-dominated sorting considers the rank determined by being in the ith front and the crowding distance to achieve a good diversity when converging.

**NSGA-III** [2][3]: A referenced based algorithm used to solve many-objective problems. The survival selected uses the perpendicular distance to the reference directions. As normalization the boundary intersection method is used [5].

## Methods

**Simulated Binary Crossover** [4]: This crossover simulates a single-point crossover in a binary representation by using an exponential distribution for real values. The polynomial mutation is defined accordingly which performs basically a binary bitflip for real numbers.



# Structure




# References


[1] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan. 2002. A fast and elitist multiobjective genetic algorithm: NSGA-II. Trans. Evol. Comp 6, 2 (April 2002), 182-197.

[2] K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints," in IEEE Transactions on Evolutionary Computation, vol. 18, no. 4, pp. 577-601, Aug. 2014.
doi: 10.1109/TEVC.2013.2281535

[3] H. Jain and K. Deb. An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point Based Nondominated Sorting Approach, Part II: Handling Constraints and Extending to an Adaptive Approach. IEEE Trans. Evolutionary Computation 18(4): 602-622 (2014)

[4] Kalyanmoy Deb, Karthik Sindhya, and Tatsuya Okabe. 2007. Self-adaptive simulated binary crossover for real-parameter optimization. In Proceedings of the 9th annual conference on Genetic and evolutionary computation (GECCO '07). ACM, New York, NY, USA, 1187-1194.

[5] Indraneel Das and J. E. Dennis. 1998. Normal-Boundary Intersection: A New Method for Generating the Pareto Surface in Nonlinear Multicriteria Optimization Problems. SIAM J. on Optimization 8, 3 (March 1998), 631-657.

# Contact

Feel free to contact me if you have any question:

Julian Blank (blankjul@egr.msu.edu)<br/>
Michigan State University<br/>
Computational Optimization and Innovation Laboratory (COIN)<br/>
East Lansing, MI 48824, USA <br/>

