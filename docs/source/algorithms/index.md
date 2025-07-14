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

.. _nb_algorithms:
```

# Algorithms

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. toctree::
   :hidden:
   :maxdepth: 2

   initialization
   usage
   list
   hyperparameters
   soo/ga
   soo/brkga
   soo/de
   soo/nelder
   soo/pso
   soo/pattern
   soo/es
   soo/sres
   soo/isres
   soo/cmaes
   soo/g3pcx
   soo/nrbo
   moo/nsga2
   moo/rnsga2
   moo/nsga3
   moo/unsga3
   moo/rnsga3
   moo/moead
   moo/ctaea
   moo/age
   moo/age2
   moo/rvea
   moo/sms
   moo/dnsga2
   moo/kgb
   moo/pinsga2
   moo/cmopso
   moo/mopso_cd
```

Algorithms are probably the reason why you got to know *pymoo*. You can find a variety of unconstrained and constrained single-, multi-, and many-objective optimization algorithms. Besides the availability of an algorithm, its usage is also of importance. The following tutorial pages show the different ways of initialization and running algorithms (functional, next, ask-and-tell) and all algorithms available in pymoo.

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. admonition:: Overview
    :class: myOwnStyle
    
    - :ref:`Initialization <nb_algorithms_init>`: How to initialize an algorithm to be run on a problem.
    - :ref:`Usage <nb_algorithms_usage>`: Different ways to run algorithms with different levels of control during optimization.
    - :ref:`List of Algorithms <nb_algorithms_list>`: Unconstrained and constrained single-, multi-, and many-objective optimization algorithms
```
