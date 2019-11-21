Versions
==============================================================================


.. _version_0_3_3:
* 0.3.3

  - New Algorithm: CMAES (by making use of the implementation published by the author)
  - New Termination Criterion: Stop an algorithm based on time 
  - New Display: Now based on a class and easy to modify
  - New Callback: Now based on a class and data during the run can be stored or the algorithm object can be modified inplace.


.. _version_0_3_2:
* 0.3.2

  - New Algorithm: Nelder Mead with box constraint handling in the design space
  - New Performance indicator: Karush Kuhn Tucker Proximity Measure (KKTPM)
  - Added Tutorial: Equality constraint handling through customized repair
  - Added Tutorial: Subset selection through GAs
  - Added Tutorial: How to use custom variables 
  - Bugfix: No pf given for problem, no feasible solutions found
  
.. _version_0_3_1:
* 0.3.1 [`Documentation <https://www.egr.msu.edu/coinlab/blankjul/pymoo-0.3.1-doc.zip>`_]

  - Merging pymop into pymoo - all test problems are included
  - Improved Getting Started Guide
  - Added Visualization
  - Added Decision Making
  - Added GD+ and IGD+
  - New Termination Criteria "x_tol" and "f_tol"
  - Added Mixed Variable Operators and Tutorial
  - Refactored Float to Integer Operators
  - Fixed NSGA-III Normalization Variable Swap
  - Fixed casting issue with latest NumPy version for integer operators
  - Removed the dependency of Cython for installation (.c files are delivered now)


.. _version_0_3_0:
* 0.3.0 

  - New documentation and global interface
  - New crossovers: Point, HUX
  - Improved version of DE
  - New Factory Methods

.. _version_0_2_2:
* 0.2.2

  - Several improvements in the code structure
  - Make the cython support optional
  - Modifications for pymop 0.2.3

.. _version_0_2_1:
* 0.2.1

  - First official release providing NSGA2, NSGA3 and RNSGA3

