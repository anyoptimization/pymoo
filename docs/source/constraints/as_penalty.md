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

.. _nb_constraints_penalty:
```

```{code-cell} ipython3
%%capture
%run ./index.ipynb
```

## Constraint Violation (CV) as Penalty

+++

Another well-known way of handling constraints is removing the constraint and adding it as a penalty to the objective(s). One easy way of achieving that is redefining the problem, as shown below using the `ConstraintsAsPenalty` class. Nevertheless, whenever two numbers are added, normalization can become an issue. Thus, commonly a penalty coefficient (here `penalty`) needs to be defined. It can be helpful to play with this parameter if the results are not satisfying.

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.constraints.as_penalty import ConstraintsAsPenalty
from pymoo.optimize import minimize
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
from pymoo.core.problem import ElementwiseProblem

class ConstrainedProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=1, n_eq_constr=0, xl=0, xu=2, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x[0] ** 2 + x[1] ** 2
        out["G"] = 1.0 - (x[0] + x[1])

problem = ConstrainedProblem()

algorithm = DE()

res = minimize(ConstraintsAsPenalty(problem, penalty=100.0),
               algorithm,
               seed=1,
               verbose=False)

res = Evaluator().eval(problem, Individual(X=res.X))

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))
```

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_constraints_no_feas_found:
```

### Solution only almost feasible

+++

Please note that this approach might not always find a feasible solution (because the algorithm does not know anything about whether a solution is feasible or not). For instance, see the example below:

```{code-cell} ipython3
res = minimize(ConstraintsAsPenalty(problem, penalty=2.0),
               algorithm,
               seed=1,
               verbose=False)

res = Evaluator().eval(problem, Individual(X=res.X))

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))
```

In such cases, it can be helpful to perform another search for the solution found to the original problem to find a feasible one. This second search method can, for instance, be realized by a local search or by using again a population-based method injecting the solution found before. Here, we demonstrate the latter:

```{code-cell} ipython3
from pymoo.operators.sampling.lhs import LHS

sampling = LHS().do(problem, 100)
sampling[0].X = res.X

algorithm = DE(sampling=sampling)

res = minimize(problem, algorithm)

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))
```
