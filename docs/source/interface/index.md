---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_interface:
```

## Interface

```{raw-cell}
:raw_mimetype: text/restructuredtext


.. toctree::
   :maxdepth: 1
   :hidden:
   
   minimize.ipynb
   problem.ipynb
   algorithm.ipynb
   termination.ipynb
   callback.ipynb
   display.ipynb
   result.ipynb
   
```

The functional interface of *pymoo* is based on a method called `minimize`, which abstracts any kind of optimization procedure in the framework. Each component of the functional interface is described in detail in the following. The guide starts with the `minimize` function itself, continues with the required and optional parameters, and ends with the `Result` object. 

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. admonition:: Overview
    :class: myOwnStyle

    - `minimize <minimize.ipynb>`_: This is the functional interface to optimize any kind of problem. The function itself has two positional parameters, problem and algorithm, and a few more optional parameters.
    - `Problem <problem.ipynb>`_: A problem object defining what to be optimized. 
    - `Algorithm <algorithm.ipynb>`_: The algorithm which shall be used for optimization. Make sure to choose a suitable algorithm for your optimization problem to ensure efficient convergence. 
    - `Termination Criterion <termination.ipynb>`_: When the algorithm should be considered as terminated. The termination can be simply based on the algorithm's number of iterations, an upper bound of function evaluations, or more sophisticated procedures.
    - `Callback <callback.ipynb>`_: How to access intermediate result during optimization to keep track of the algorithm itself or modify attributes of the algorithm dynamically.
    - `Display <display.ipynb>`_: When `verbose=True`, then the algorithm prints out some information in each iteration. The printout is different depending on if it is a single or multi-objective optimization problem and if the Pareto-front is known or unknown.
    - `Result <result.ipynb>`_: The result object being returned by the minimize method. Access to the optimum/optima found and some more information such as the running time or even the whole algorithm's run history.
```

Please note that besides the **functional** interface, *pymoo* also has an **object-oriented** interface (more information [here](../algorithms/index.ipynb)) for running an optimization algorithm. Both have their upsides and downsides, and one can be more convenient to use than another in specific situations. The functional interface allows optimization just in a few lines; the object-oriented interface allows quickly to alter an existing algorithm's behavior.
