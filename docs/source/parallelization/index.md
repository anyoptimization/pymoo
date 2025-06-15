---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  name: ''
  display_name: ''
---

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_parallelization:
```

## Parallelization

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. toctree::
   :hidden:
   :maxdepth: 1
   
   vectorized.ipynb
   starmap.ipynb
   joblib.ipynb
   gpu.ipynb
   custom.ipynb
```

In practice, parallelization is essential and can significantly speed up optimization. 
For population-based algorithms, the evaluation of a set of solutions can be parallelized easily 
by parallelizing the evaluation itself.

This section covers various parallelization strategies available in *pymoo*:

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. admonition:: Overview
    :class: myOwnStyle

    - `Vectorized Operations <vectorized.ipynb>`_: Using NumPy matrix operations for efficient parallel computation
    - `Starmap Interface <starmap.ipynb>`_: Using Python's multiprocessing starmap for threads and processes
    - `Joblib <joblib.ipynb>`_: Advanced parallelization with joblib's flexible backend system
    - `GPU Acceleration <gpu.ipynb>`_: High-performance computing using CUDA and PyTorch
    - `Custom Parallelization <custom.ipynb>`_: Implementing your own parallelization strategy
```
