---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: default
  language: python
  name: default
---

# FAQ

+++

Below you will find some answers to frequently asked questions from the past:

+++

**Q1: When I enable** `verbose=True`, **what does the output mean?** 

The output differs depending on the type of problem and if the Pareto-front is known or not. Please have a look at the description of each column at the [Display](interface/display.ipynb) guide.

+++

**Q2: How can equality constraints be handled?** 

Genetic algorithms are not able to deal with *equality* constraints out of the box. Nevertheless, modifying the search space to always satisfy the constraints can make evolutionary operators work in your favor. Another approach is to add a [Repair](operators/repair.ipynb) operator to find a feasible solution close by to an existing one.
