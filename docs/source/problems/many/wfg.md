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

.. _nb_wfg:
```

# WFG

WFG1 to WFG9 are available.

```{code-cell} ipython3
from pymoo.problems.many.wfg import WFG1
wfg = WFG1(n_var=10, n_obj=3)
```
