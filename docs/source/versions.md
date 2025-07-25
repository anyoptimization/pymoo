---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_version:
```

# Versions

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. tip::
    To access deprecated documentations, please use the following credentials:

    Username: pymoo, Password: pymoo

    (The access is protected to avoid search engines directing to deprecated documentations)
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _version_0_6_1_5:
```

#### 0.6.1.5 [[Documentation](http://archive.pymoo.org/0.6.1.5/)]

- Additional bug fixes and stability improvements
- Enhanced compatibility across different environments
- Performance optimizations

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _version_0_6_1_3:
```

#### 0.6.1.3 [[Documentation](http://archive.pymoo.org/0.6.1.3/)]

- Compatibility with Numpy 2.0
- Make Autograd for Automatic Differentiation Optional
- Incorporate all Bug Fixes from Pull Requests

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _version_0_6_1:
```

#### 0.6.1 [[Documentation](http://archive.pymoo.org/0.6.1/)]

- Minor changes and bugfixes that have been reported
- Added KGB for dynamic optimization

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _version_0_6_0:
```

#### 0.6.0 [[Documentation](http://archive.pymoo.org/0.6.0/)]

- Breaking changes: Factory methods have been deprecated or deactivated (because of maintenance overhead and hiding of constructor parameters)
- New Problems: DF (for dynamic optimization)
- New Algorithms: G3-PCX, SMS-EMOA, RVEA, AGE-MOEA2, DNSGA2
- Mixed Variable Optimization: Improved support for mixed variable optimization.
- Hyperparameter Tuning: Basic interface for hyperparameter tuning
- Constrained Handling: Improved tutorial explaining how constrained can be handled within different kinds of algorithms
- New Termination Criteria: The implementation now requires returning a floating point number. It is initialized by zero, and a one indicates the algorithm has terminated. This also allows activating a progress bar.
- New Parallelization: The interface has been changed and a class for running a parallel evaluation has been defined.

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _version_0_5_0:
```

#### 0.5.0 [[Documentation](http://archive.pymoo.org/0.5.0/)]

- New Theme: As you might have noticed, *pymoo* got a new HTML theme, responsive, and has a better navigation bar.
- New Project Structure: This includes some breaking changes. Now, the algorithms are grouped into different categories. For instance, `NSGA2` is now available at `pymoo.algorithms.moo.NSGA2`. 
- New Algorithms: RVEA, AGEMOEA, ES, SRES, ISRES
- New Problem Implementation: The new version distinguishes between a `Problem` and an `ElementwiseProblem`. This has the advantage of handling the two different implementations on an object level.
- New Interface: Most algorithms follow the `infill` and `advance` schema, which makes it very simple to write a for loop-based approach and customizing the algorithm's default behavior (for instance, a local search)
- New Getting Started Guide consisting of five parts explaining better how *pymoo* can be used. The different alternatives of defining a problem and running an algorithm have been outsourced to the corresponding tutorial pages.

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _version_0_4_2:
```

#### 0.4.2 [[Documentation](http://archive.pymoo.org/0.4.2/)]

- Improved Getting Started Guide with a new interface of providing functions instead of implementing the problem class
- New Algorithm: PSO for single-objective problems
- New Loop-wise Execution: The algorithm object can be used directly by calling its next method
- New Tutorial: An implementation of checkpoints to resume runs
- New Test Problems Suites (Constrained): DAS-CMOP and MW (contributed by cyrilpic)
- New Operators for Permutations: OrderCrossover and InversionMutation and usage to optimize routes for the TSP and Flowshop problem (contributed by Peng-YM )
- New Crossover: Parent Centric Crossover (PCX) which is known to work well on problems where some variables have dependencies on each other
- Bugfix: Remove evaluation calls in Problem class during print

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _version_0_4_1:
```

#### 0.4.1 [[Documentation](http://archive.pymoo.org/0.4.1/)]

- New Feature: Riesz s-Energy Method to generate a well-spaced point-set on the unit simplex (reference directions) of arbitrary size.
- New Algorithm: An implementation of Hooke and Jeeves Pattern Search (well-known single-objective algorithm)
- New Documentation: We have re-arranged the documentation and explain now the minimize interface in more detail.
- New Feature: The problem can be parallelized by directly providing a starmapping callable (Contribution by Josh Karpel).
- Bugfix: MultiLayerReferenceDirectionFactory did not work because the scaling was disabled.

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _version_0_4_0:
```

#### 0.4.0 [[Documentation](http://archive.pymoo.org/0.4.0/)]

  - New Algorithm: CMA-ES (Implementation published by the Author)
  - New Algorithm: Biased-Random Key Genetic Algorithm (BRKGA)
  - New Test Problems: WFG
  - New Termination Criterion: Stop an Algorithm based on Time
  - New Termination Criterion: Objective Space Tolerance for Multi-objective Problems
  - New Display: Easily modify the Printout in each Generation
  - New Callback: Based on a class now to allow to store data in the object.
  - New Visualization: Videos can be recorded to follow the algorithm's progress.
  - Bugfix: NDScatter Plot
  - Bugfix: Hypervolume Calculations (Vendor Library)

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _version_0_3_2:
```

#### 0.3.2 [[Documentation](http://archive.pymoo.org/0.3.2/)]

  - New Algorithm: Nelder Mead with box constraint handling in the design space
  - New Performance indicator: Karush Kuhn Tucker Proximity Measure (KKTPM)
  - Added Tutorial: Equality constraint handling through customized repair
  - Added Tutorial: Subset selection through GAs
  - Added Tutorial: How to use custom variables 
  - Bugfix: No pf given for problem, no feasible solutions found

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _version_0_3_1:
```

#### 0.3.1 [[Documentation](http://archive.pymoo.org/0.3.1/)]

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

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _version_0_3_0:
```

#### 0.3.0

  - New documentation and global interface
  - New crossovers: Point, HUX
  - Improved version of DE
  - New Factory Methods

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _version_0_2_2:
```

#### 0.2.2

  - Several improvements in the code structure
  - Make the cython support optional
  - Modifications for pymop 0.2.3

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _version_0_2_1:
```

#### 0.2.1

  - First official release providing NSGA2, NSGA3 and RNSGA3
