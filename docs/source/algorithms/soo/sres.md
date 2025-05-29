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

.. _nb_sres:
```

.. meta::
   :description: The stochastic ranking is based on bubble sort and provides infeasible solutions a chance to survive during the environment selection. Adding this selection to an evolution strategy method has shown to be an effective optimization method: Stochastic Ranking Evolutionary Strategy.

+++

.. meta::
   :keywords: Stochastic Ranking Evolutionary Strategy, SRES,  Constrained Optimization, Real-Valued Optimization, Single-objective Optimization, Python

+++

# SRES: Stochastic Ranking Evolutionary Strategy

+++

Many different constraint handling methods have been proposed in the past. One way of addressing constraints in evolutionary strategy is to change the selection operator and give infeasible solutions a chance to survive. 
The survival is based on stochastic ranking, and thus the method is known as Stochastic Ranking Evolutionary Strategy <cite data-cite="sres"></cite>. 

The stochastic ranking is proposed as follows:

+++

<div style="text-align: center;">
    <img src="https://github.com/anyoptimization/pymoo-data/blob/main/docs/images/sr.png?raw=true" width="450">
</div>

+++

Together with the effective evolutionary strategy search algorithm, this provides a powerful method to optimize constrained problems. 

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("g1")

algorithm = SRES(n_offsprings=200, rule=1.0 / 7.0, gamma=0.85, alpha=0.2)

res = minimize(problem,
               algorithm,
               ("n_gen", 200),
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))
```

An improved version of SRES, called ISRES, has been proposed to deal with dependent variables. The dependence has been addressed by using the differential between individuals as an alternative mutation.

+++

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.soo.nonconvex.sres.SRES
    :noindex:
```
