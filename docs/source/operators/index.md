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

.. _nb_operators:
```

## Operators

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. toctree::
   :maxdepth: 1
   :hidden:

   sampling
   selection
   crossover
   mutation
   repair
   survival
```



Operators are the key to customize genetic algorithms. In the following, the different types of operators are listed. For details about each operator, we refer to our corresponding documentation.

+++

### Sampling

+++

|Name|Convenience|
|---|---|
|[Random](sampling.ipynb)|"(real\|int\|real)_random"|
|[Latin Hypercube Sampling](sampling.ipynb)|"real_lhs"|
|[Random Permutation Sampling](sampling.ipynb)|"perm_random"|

+++

### Selection

+++

|Name|Convenience|
|---|---|
|[Random](selection.ipynb)|"random"|
|[Tournament Selection](selection.ipynb)|"tournament"|

+++

### Mutation

+++

|Name|Convenience|
|---|---|
|[Polynomial](mutation.ipynb)|"(real\|int)_pm"|
|[Bitflip](mutation.ipynb)|"bin_bitflip"|
|[Inverse Mutation](mutation.ipynb)|"perm_inv"|

+++

### Crossover

+++

|Name|Convenience|
|---|---|
|[Simulated Binary](crossover.ipynb)|"(real\|int)_sbx"|
|[Uniform](crossover.ipynb)|"(real\|bin\|int)_ux"|
|[Half Uniform](crossover.ipynb)|"(bin\|int)_hux"|
|[Differential Evolution](crossover.ipynb)|"real_de"|
|[One Point](crossover.ipynb)|"(real\|int\|real)_one_point"|
|[Two Point](crossover.ipynb)|"(real\|int\|real)_two_point"|
|[K Point](crossover.ipynb)|"(real\|int\|real)_k_point"|
|[Exponential](crossover.ipynb)|"(real\|bin\|int)_exp"|
|[Order Crossover](crossover.ipynb)|"perm_ox"|
|[Edge Recombination Crossover](crossover.ipynb)|"perm_erx"|
