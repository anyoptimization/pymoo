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

.. _nb_constraints_as_obj:
```

```{code-cell} ipython3
from pymoo.core.problem import ElementwiseProblem

class ConstrainedProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=1, n_eq_constr=0, xl=0, xu=2, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x[0] ** 2 + x[1] ** 2
        out["G"] = 1.0 - (x[0] + x[1])
```

### Feasibility First (Parameter-less Approach)

+++

Most algorithms follow the so-called feasibility first approach (also known as a parameter-less approach). There, the constraint violation measure is chosen from a practical standpoint. In order to evaluate a solution, any designer will first check if the solution is feasible. Suppose the solution is infeasible (that is, at least one constraint is violated). In that case, the designer will never bother to compute its objective function value (such as the cost of the design). It does not make sense to compute the objective function value of an infeasible solution because the solution simply cannot be implemented in practice. Motivated by this argument, we devise the following penalty term where infeasible solutions are compared based on only their constraint violation values:

<div style="text-align: center;">
    <img src="https://github.com/anyoptimization/pymoo-data/blob/main/docs/images/parameter_less.png?raw=true" width="350">
</div>


This constraint handling method is straightforward to integrate into a framework because it applies to many different algorithms that are either based on sorting or use the comparison of solutions. For more details please consult <cite data-cite="parameter-less"></cite>.

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

problem = ConstrainedProblem()

algorithm = GA(pop_size=100)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))
```
