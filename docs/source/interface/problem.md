---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

+++ {"raw_mimetype": "text/restructuredtext"}

.. _nb_interface_problem:

+++

## Problem


+++

There exist a couple of different ways for defining an optimization problem in *pymoo*. In contrast to other optimization frameworks in Python, the preferred way is to define an **object**. However, a problem can also be defined by functions as shown [here](../problems/index.ipynb). Most algorithms in *pymoo* are population-based, which implies that in each generation, not a single but multiple solutions are evaluated. Thus, the problem implementation retrieves the set of solutions to provide the most flexibility to the end-user. This flexibility allows you to implement a custom parallelization and thus to use your hardware most efficiently. Three different ways of defining a problem are shown below:

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. admonition:: Overview
    :class: myOwnStyle

    - `Problem <../problems/index.ipynb>`_: Object-oriented definition :code:`Problem` which implements a method evaluating a **set** of solutions.
    - `ElementwiseProblem <../problems/index.ipynb>`_: Object-oriented definition :code:`ElementwiseProblem` which implements a function evaluating a **single** solution at a time. 
    - `FunctionalProblem <../problems/index.ipynb>`_: Define a problem :code:`FunctionalProblem` by using a **function** for each objective and constraint.
```

Next, we define an **unconstrained** optimization problem with two variables and two objectives. Because the lower and upper bounds are identical for both variables, only a float value is passed to the `Problem` constructor. Assuming the `Algorithm` has a population size N, the input variable `x` is a two-dimensional matrix with the dimensions (N,2). The input has two columns because the optimization problem has `n_var=2`. Thus, to evaluate the problem makes use of the vectorized calculations `[:, 0]` and `[:, 1]` to select the first and second variables for each row in the input matrix `x`.

```{code-cell} ipython3
import numpy as np
from pymoo.core.problem import Problem

class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2, 
                         n_obj=2, 
                         xl=-2.0, 
                         xu=2.0)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[:, 0]**2 + x[:, 1]**2)
        f2 = (x[:, 0]-1)**2 + x[:, 1]**2
        out["F"] = np.column_stack([f1, f2])
 
```

Below, we define a **constrained** optimization problem with two variables and two objectives. Here, the problem is defined **element-wise**. The lower and upper bounds, `xl` and `xu`, are defined using a vector with a length equal to the number of variables. The input `x` is a **one-dimensional** array of length two and is called N times in each iteration for the algorithm discussed above.

```{code-cell} ipython3
import numpy as np
from pymoo.core.problem import ElementwiseProblem

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2, 
                         n_obj=2, 
                         n_ieq_constr=2, 
                         xl=np.array([-2,-2]), 
                         xu=np.array([2,2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2
        
        g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8
        
        out["F"] = [f1, f2]
        out["G"] = [g1, g2]
        

problem = MyProblem()
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. admonition:: Tip
    :class: myOwnStyle

    For more information, please look at the problem `tutorial <../problems/index.ipynb>`_. Moreover, a number of test problems frequently being use for benchmarking the performance of an algorithm are listed `here <../problems/test_problems.ipynb>`_.
```
