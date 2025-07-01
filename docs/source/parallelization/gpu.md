---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_parallelization_gpu:
```

# GPU Acceleration

If the problem evaluation takes a lot of time, we can optimize above vectorized matrix operation by adopting GPU acceleration. The modern GPU matrix manipulation framework such as PyTorch or JAX makes it easy.

+++

## PyTorch

The problem is evaluated using PyTorch framework should follow below steps:

1. Converts numpy vectorized matrix to tensor and copy the data to cuda device
2. Calculates the problem using tensor
3. Returns the final results and copy to CPU so that pymoo will schedule it to next iteration.

```{code-cell} ipython3
import numpy as np
import torch
from pymoo.core.problem import Problem

class MyProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=0, xl=-5, xu=5, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
         x = torch.from_numpy(x)
         if torch.cuda.is_available():
             x = x.cuda()
         f = torch.sum(torch.pow(x, 2), dim=1)
         out["F"] = f.detach().cpu().clone().numpy()

problem = MyProblem()
```

## JAX

JAX as accelerated numpy and it provides a numpy-inspired interface for convenience. By default JAX executes operations one at a time, in sequence. Using a just-in-time (JIT) compilation decorator, sequences of operations can be optimized together and run at once. In order to apply JIT compilation decorator, some private helper functions `_eval_F` and `_eval_G` are wrapped.

**IMPORTANT:** user should turn on float64 configuration if the problem's dtype is float64, otherwise some precision may lose and the result may be different.

```{code-cell} ipython3
import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
from pymoo.core.problem import Problem

jax.config.update("jax_enable_x64", True) # default is float32 
jax.config.update('jax_disable_jit', False) # for debugging

class MyProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=0, xl=-50, xu=50, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        _x = jnp.array(x)
        f = self._eval_F(_x)
        out["F"] = np.asarray(f)

    @partial(jax.jit, static_argnums=0)
    def _eval_F(self, x):
        return jnp.sum(jnp.power(x, 2), axis=1)
    
problem = MyProblem()
```
