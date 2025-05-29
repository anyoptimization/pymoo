---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

+++ {"raw_mimetype": "text/restructuredtext"}

.. _nb_interface_minimize:

+++

## Minimize


+++

The `minimize` function provides the external interface for any kind of optimization to be performed. The minimize method arguments and options look as follows:

+++

```python
def minimize(problem,
             algorithm,
             termination=None,
             seed=None,
             verbose=False,
             display=None,
             callback=None,
             return_least_infeasible=False,
             save_history=False
             )
```

+++

- `problem`: A [Problem](../problems/index.ipynb)  object that contains the problem to be solved.
- `algorithm`: An [Algorithm](../algorithms/index.ipynb) objective which represents the algorithm to be used.
- `termination`: A [Termination](termination.ipynb) object or a tuple defining when the algorithm has terminated. If not provided, a default termination criterion will be used. Purposefully, we list the *termination* as a parameter and not an option. Specific algorithms might need some refinement of the termination to work reliably. 
- `seed`: Most algorithms underly some randomness. Setting the *seed* to a positive integer value ensures reproducible results. If not provided, a random seed will be set automatically, and the used integer will be stored in the [Result](result.ipynb) object.
- `verbose`: Boolean value defining whether the output should be printed during the run or not.
- `display`: You can overwrite what output is supposed to be printed in each iteration. Therefore, a custom [Display](display.ipynb) object can be used for customization purposes. 
- `save_history`: A boolean value representing whether a snapshot of the algorithm should be stored in each iteration. If enabled, the [Result](result.ipynb) object contains the history.
- `return_least_infeasible`: Whether if the algorithm can not find a feasible solution, the least infeasible solution should be returned. By default `False`.

+++

Note, the `minimize` function creates a **deep copy** of the algorithm object before the run.
This ensures that two independent runs with the same algorithm and same random seed have the same results without any side effects. However, to access the algorithm's internals, you can access the object being used by `res.algorithm` where `res` is an instance of the [Result](result.ipynb) object.

+++

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autofunction:: pymoo.optimize.minimize
```
