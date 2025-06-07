---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_kgb:
```

# KGB-DMOEA: Knowledge-Guided Bayesian Dynamic Multi-Objective Evolutionary Algorithm

+++

KGB-DMOEA is a sophisticated evolutionary algorithm for dynamic multi-objective optimization problems (DMOPs). It employs a knowledge-guided Bayesian classification approach to adeptly navigate and adapt to changing Pareto-optimal solutions in dynamic environments. This algorithm utilizes past search experiences, distinguishing them as beneficial or non-beneficial, to effectively direct the search in new scenarios.

+++

### Key Features


- **Knowledge Reconstruction-Examination (KRE):** Dynamically re-evaluates historical optimal solutions based on their relevance and utility in the current environment. 
- **Bayesian Classification:** Employs a Naive Bayesian Classifier to forecast high-quality initial populations for new environments.
- **Adaptive Strategy:** Incorporates dynamic parameter adjustment for optimized performance across varying dynamic contexts.

```{code-cell} ipython3
from pymoo.algorithms.moo.kgb import KGB
from pymoo.core.callback import CallbackCollection
from pymoo.optimize import minimize
from pymoo.problems.dyn import TimeSimulation
from pymoo.problems.dynamic.df import DF1

from pymoo.visualization.video.callback_video import ObjectiveSpaceAnimation

problem = DF1(taut=2, n_var=2)

algorithm = KGB()

res = minimize(problem,
               algorithm,
               termination=('n_gen', 10),
               callback=TimeSimulation(),
               seed=1,
               verbose=False)
```

### Parameters 

- **perc_detect_change (float, optional):** Proportion of the population used to detect environmental changes. 
- **perc_diversity (float, optional):** Proportion of the population allocated for introducing diversity. 
- **c_size (int, optional):** Cluster size.
- **eps (float, optional):** Threshold for detecting changes. Default: 
- **perturb_dev (float, optional):** Deviation for perturbation in diversity introduction. 

+++

### References

```{raw-cell}
Yulong Ye, Lingjie Li, Qiuzhen Lin, Ka-Chun Wong, Jianqiang Li, Zhong Ming. “A knowledge guided Bayesian classification for dynamic multi-objective optimization”. Knowledge-Based Systems, Volume 251, 2022.
```
