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

.. _nb_constraints_eps:
```

```{code-cell} ipython3
from pymoo.core.problem import ElementwiseProblem

class ConstrainedProblemWithEquality(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=1, n_eq_constr=1, xl=0, xu=1, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x[0] + x[1]
        out["G"] = 1.0 - (x[0] + x[1])
        out["H"] = 3 * x[0] - x[1]
```

### $\epsilon$-Constraint Handling

+++

Instead of directly redefining the problem, one can also redefine an algorithm that changes its conclusion, whether a solution is feasible given the constraint violation over time. One common way is to allow an $\epsilon$ amount of infeasibility to still consider a solution as feasible. Now, one can decrease the $\epsilon$ over time and thus finally fall back to a feasibility first algorithm. The $\epsilon$  has reached zero depending on `perc_eps_until`. For example,  if `perc_eps_until=0.5` then after 50\% of the run has been completed $\epsilon=0$.

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. admonition:: Info
    :class: myOwnStyle

    This constraint handling method has been added recently and is still experimental. Please let us know if it has or has not worked for your problem.
```

Such a method can be especially useful for equality constraints which are difficult to satisfy. See the example below.

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.optimize import minimize
from pymoo.problems.single import G1

problem = ConstrainedProblemWithEquality()

algorithm = AdaptiveEpsilonConstraintHandling(DE(), perc_eps_until=0.5)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))
```
