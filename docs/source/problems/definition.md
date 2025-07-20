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

.. _nb_problem_definition:
```

# Definition

+++

Problems have to be defined, and some information has to be provided.
In contrast to other frameworks, we do not share the opinion that just defining a problem by a function is the most convenient one.
In `pymoo` the problem is defined by an **object** that contains some metadata, for instance the number of objectives, constraints, lower and upper bounds in the design space. These attributes are supposed to be defined in the constructor by overriding the `__init__` method.

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. csv-table:: Types of Output
   :header: "Argument", "Description"
   :widths: 5, 30

   :code:`n_var`, "Integer value representing the number of design variables."
   :code:`n_obj`, "Integer value representing the number of objectives."
   :code:`n_constr`, "Integer value representing the number of constraints."
   :code:`xl`,  "Float or :code:`np.ndarray` of length :code:`n_var` representing the lower bounds of the design variables."
   :code:`xu`,  "Float or :code:`np.ndarray` of length :code:`n_var` representing the upper bounds of the design variables."
   :code:`vtype`, "(optional) A type hint for the user what variable should be optimized."
```

Moreover, in *pymoo* there exist three different ways for defining a problem:

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. admonition:: Overview
    :class: myOwnStyle

    - `Problem <#nb_problem_definition_vectorized>`_: Object-oriented definition :code:`Problem` which implements a method evaluating a **set** of solutions.
    - `ElementwiseProblem <#nb_problem_definition_elementwise>`_: Object-oriented definition :code:`ElementwiseProblem` which implements a function evaluating a **single** solution at a time. 
    - `FunctionalProblem <#nb_problem_definition_functional>`_: Define a problem :code:`FunctionalProblem` by using a **function** for each objective and constraint.
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_problem_definition_vectorized:
```

## Problem (vectorized)

+++

The majority of optimization algorithms implemented in *pymoo* are population-based, which means that more than one solution is evaluated in each generation. This is ideal for implementing a parallelization of function evaluations. Thus, the default definition of a problem retrieves a **set** of solutions to be evaluated. The actual function evaluation takes place in the `_evaluate` method, which aims to fill the `out` dictionary with the corresponding data. 
The function values are supposed to be written into `out["F"]` and the constraints into `out["G"]` if `n_constr` is greater than zero. If another approach is used to compute the function values or the constraints, they must be appropriately converted into a two-dimensional `numpy` array (one dimension for the function values, the other dimension for each element of the population evaluated in the current round). For example, if the function values are written in a regular python list like `F_list = [[<func values for individual 1>], [<func values for individual 2>], ...]`, before returning from the `_evaluate` method, the list must be converted to numpy array with `out["F"] = np.row_stack(F_list_of_lists)`. 

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. admonition:: Tip
    :class: myOwnStyle

    How the objective and constraint values are calculate is **irrelevant** from a pymoo's point of view. Whether it is a simple mathematical equation or a discrete-event simulation, you only have to ensure that for each input the corresponding values have been set.
```

The example below shows a modified **Sphere** problem with a radial constraint located at the center. The problem consists of 10 design variables, one objective, one constraint, and the lower and upper bounds of each variable are in the range of 0 and 1. 

```{code-cell} ipython3
import numpy as np
from pymoo.core.problem import Problem


class SphereWithConstraint(Problem):

    def __init__(self):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=1, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum((x - 0.5) ** 2, axis=1)
        out["G"] = 0.1 - out["F"]
```

Assuming the algorithm being used requests to evaluate a set of solutions of size 100, then the input NumPy matrix `x`  will be of the shape `(100,10)`. Please note that the two-dimensional matrix is summed up on the first axis which results in a vector of length 100 for `out["F"]`. Thus, NumPy performs a vectorized operation on a matrix to speed up the evaluation.

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_problem_definition_elementwise:
```

## ElementwiseProblem (loop)

```{code-cell} ipython3
import numpy as np
from pymoo.core.problem import ElementwiseProblem


class ElementwiseSphereWithConstraint(ElementwiseProblem):

    def __init__(self):
        xl = np.zeros(10)
        xl[0] = -5.0
        
        xu = np.ones(10)
        xu[0] = 5.0
        
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=2, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum((x - 0.5) ** 2)
        out["G"] = np.column_stack([0.1 - out["F"], out["F"] - 0.5])
```

Regardless of the number of solutions being asked to be evaluated, the `_evaluate` function retrieves a vector of length 10. The `_evaluate`, however, will be called for each solution. Implementing an element-wise problem, the [Parallelization](../parallelization/index.ipynb) available in *pymoo* using processes or threads can be directly used.
Moreover, note that the problem above uses a vector definition for the lower and upper bounds (`xl` and `xu`) because the first variables should cover a different range of values.

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_problem_definition_functional:
```

## FunctionalProblem (loop)

+++

Another way of defining a problem is through functions. On the one hand, many function calls need to be performed to evaluate a set of solutions, but on the other hand, it is a very intuitive way of defining a problem.

```{code-cell} ipython3
import numpy as np
from pymoo.problems.functional import FunctionalProblem


objs = [
    lambda x: np.sum((x - 2) ** 2),
    lambda x: np.sum((x + 2) ** 2)
]

constr_ieq = [
    lambda x: np.sum((x - 1) ** 2)
]

n_var = 10

problem = FunctionalProblem(n_var,
                            objs,
                            constr_ieq=constr_ieq,
                            xl=np.array([-10, -5, -10]),
                            xu=np.array([10, 5, 10])
                            )

F, G = problem.evaluate(np.random.rand(3, 10))

print(f"F: {F}\n")
print(f"G: {G}\n")
```

## Add Known Optima

+++

If the optimum for a problem is known, this can be directly defined in the `Problem` class. Below, an example shows the test problem `ZDT1` where the Pareto-front has been analytically derived and discussed in the paper. Thus, the `_calc_pareto_front` method returns the Pareto-front.

```{code-cell} ipython3
class ZDT1(Problem):
    
    def __init__(self, n_var=30, **kwargs):
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=0, xl=0, xu=1, vtype=float, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:, 1:], axis=1)
        f2 = g * (1 - np.power((f1 / g), 0.5))

        out["F"] = np.column_stack([f1, f2])
```

## API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. class:: pymoo.core.problem.Problem

   .. automethod:: evaluate(X)
   .. automethod:: pareto_front(X)
```
