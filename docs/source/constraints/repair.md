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

.. _nb_repair:
```

```{code-cell} ipython3
%%capture
%run ./index.ipynb
```

## Repair Operator 

+++

A simple approach is to handle constraints through a repair function. This is only possible if the equation of the constraint is known. The repair makes sure every solution that is evaluated is, in fact, feasible. Let us consider the equality constraint example.

+++

Let us define a `Repair` operator that always satisfies the equality constraint (the inequality constraint is simply ignored and will be figured out by the algorithm).

```{code-cell} ipython3
from pymoo.core.repair import Repair

class MyRepair(Repair):

    def _do(self, problem, X, **kwargs):
        X[:, 0] = 1/3 * X[:, 1]
        return X
```

Now the algorithm object needs to be initialized with the `Repair` operator and then can be run to solve the problem:

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

algorithm = GA(repair=MyRepair())

res = minimize(ConstrainedProblemWithEquality(),
               algorithm,
               ('n_gen', 20),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))
```

If you would like to compare the solution without a repair you will see how searching only in the feasible space helps:

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize


algorithm = GA()

res = minimize(ConstrainedProblemWithEquality(),
               algorithm,
               ('n_gen', 20),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))
```
