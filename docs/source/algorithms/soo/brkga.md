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

.. _nb_brkga:
```

# BRKGA: Biased Random Key Genetic Algorithm


+++

An excellent and very informative presentation about BRKGAs can be found [here](http://mauricio.resende.info/talks/2012-09-CLAIO2012-brkga-tutorial-both-days.pdf). BRKGAs are known to perform well on combinatorial problems.

+++

<div style="text-align: center;">
    <img src="https://github.com/anyoptimization/pymoo-data/blob/main/docs/images/brkga.png?raw=true" width="350">
</div>

+++

Instead of customizing evolutionary operators, decoding has to be defined. Therefore, evolution takes place solely on real-valued variables. 

Let us define a permutation problem which derives an order by sorting real-valued variables:

```{code-cell} ipython3
import numpy as np

from pymoo.core.problem import ElementwiseProblem


class MyProblem(ElementwiseProblem):

    def __init__(self, my_list):
        self.correct = np.argsort(my_list)
        super().__init__(n_var=len(my_list), n_obj=1, n_ieq_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        pheno = np.argsort(x)
        out["F"] = - float((self.correct == pheno).sum())
        out["pheno"] = pheno
        out["hash"] = hash(str(pheno))
```

Since duplicate eliminates is an essential aspect for evolutionary algorithms, we have to make sure all duplicates with respect to the permutation (and not to the real values) are filtered out.

```{code-cell} ipython3
from pymoo.core.duplicate import ElementwiseDuplicateElimination


class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return a.get("hash") == b.get("hash")
```

Then, we define a problem that has to sort a list by their values.

```{code-cell} ipython3
np.random.seed(2)
list_to_sort = np.random.random(20)
problem = MyProblem(list_to_sort)
print("Sorted by", np.argsort(list_to_sort))
```

Finally, we use `BRKGA` to obtain the sorted list:

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.optimize import minimize

algorithm = BRKGA(
    n_elites=200,
    n_offsprings=700,
    n_mutants=100,
    bias=0.7,
    eliminate_duplicates=MyElementwiseDuplicateElimination())

res = minimize(problem,
               algorithm,
               ("n_gen", 75),
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
print("Solution", res.opt.get("pheno")[0])
print("Optimum ", np.argsort(list_to_sort))
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.soo.nonconvex.brkga.BRKGA
    :noindex:
```
