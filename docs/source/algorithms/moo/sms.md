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

.. _nb_sms:
```

# SMS-EMOA: Multiobjective selection based on dominated hypervolume

+++

The algorithm is implemented based on <cite data-cite="sms"></cite>. The hypervolume measure (or s-metric) is a frequently applied quality measure for comparing the results of evolutionary multiobjective optimization algorithms (EMOAs). 

+++

<div style="text-align: center;">
    <img src="https://github.com/anyoptimization/pymoo-data/blob/main/docs/images/sms.png?raw=true" width="300">
</div>

+++

SMS-EMOA aims to maximize the dominated hypervolume within the optimization process. It features a selection operator based on the hypervolume measure combined with the concept of non-dominated sorting. As a result, the algorithm’s population evolves to a well-distributed set of solutions, focusing on interesting regions of the Pareto front. 

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. admonition:: Info
    :class: myOwnStyle

    Note that the hypervolume metric becomes computationally very expensive for more than three objectives.
```

### Example

```{code-cell} ipython3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

algorithm = SMSEMOA()

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.moo.sms.SMSEMOA
    :noindex:
```
