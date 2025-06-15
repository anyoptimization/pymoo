---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
---

.. meta::
   :description: A guide which introduces the most important steps to get started with pymoo, an open-source multi-objective optimization framework in Python.

+++

.. meta::
   :keywords: Multi-objective Optimization, Python, Evolutionary Computation, Optimization Test Problem, Hypervolume

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_getting_started_preface:
```

# Preface: Basics and Challenges

+++

Without any loss of generality an optimization problem can be defined by:

+++

\begin{align}
\begin{split}
\min \quad& f_{m}(x) \quad \quad \quad \quad m = 1,..,M  \\[4pt]
\text{s.t.}   \quad& g_{j}(x) \leq 0  \quad \; \; \,  \quad j = 1,..,J \\[2pt]
\quad& h_{k}(x) = 0        \quad  \; \; \quad k = 1,..,K \\[4pt]
\quad& x_{i}^{L} \leq x_{i} \leq x_{i}^{U}  \quad i = 1,..,N \\[2pt]
\quad& x \in \Omega
\end{split}
\end{align}

+++

where $x_i$ represents the $i$-th variable to be optimized, $x_i^{L}$ and $x_i^{U}$ its lower and upper bound, $f_m$ the $m$-th objective function, $g_j$ the $j$-th inequality constraint and $h_k$ the $k$-th equality constraint.  
The objective function(s) $f$ are supposed to be minimized by satisfying all equality and inequality constraints. If a specific objective function is maximized ($\max f_i$), one can redefine the problem to minimize its negative value ($\min -f_i$).

Instead of starting coding your problem immediately, it is recommendable to first think about the mathematical problem formulation. Doing so makes you being aware of the complete optimization problem. This also helps you to identify the challenging facets of your optimization problem and, thus, to select a suitable algorithm. In this guide, we will demonstrate an example of a multi-objective problem, use `pymoo` to obtain a solution set, and theoretically derive the optimum for verification purposes.

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. tip::
    If you are new to multi-objective optimization and are not familiar with essential concepts, a look into "Multi-Objective Optimization Using Evolutionary Algorithms " by Kalyanmoy Deb might be a good starting point.
```

If you have some experience solving optimization problems, the following might already be apparent to you. However, if you are new to optimization, thinking about your problem's characteristics is vital. In the following, a few common facts and challenges to consider when intending to solve a (real-world) optimization problem are discussed:

+++

**Variable Types.** The variables span the search space $\Omega$ of your optimization problem. Thus, the type of variables is an essential aspect of the problem to be paid attention to. Different variables types, such as continuous, discrete/integer, binary, or permutation, define the characteristics of the search space. In some cases, the variable types might even be mixed, which increases the complexity further. 

+++

**Number of Variables.** Not only the type but also the number of variables ($N$) is essential. For either a very small or large number, different algorithms are known to work more efficiently. You can imagine that solving a problem with only ten variables is fundamentally different from solving one with a couple of thousand. For large-scale optimization problems, even the second-order derivative becomes computationally very expensive, and efficiently handling the memory plays a more important role.

+++

**Number of Objectives.** Some optimization problems have more than one conflicting objective ($M>1$) to be optimized. Before researchers have investigated multi-objective optimization, single-objective problems were the main focus. Single-objective optimization is only a particular case where $M=1$. In multi-objective optimization, the solution's domination relation generalizes the comparison of two scalars in single-objective optimization. 
Moreover, having more than one dimension in the objective space, the optimum (most of the time) consists of a set of non-dominated solutions. 
Because a *set* of solutions should be obtained, population-based algorithms have mainly been used as solvers.

+++

**Constraints.** Optimization problems have two types of constraints, inequality ($g$) and equality ($h$) constraints. From an end-user perspective, constraints have a priority over objective values. No matter how good the solution's objectives are, it is considered infeasible if it turns out to violate just a single constraint. Constraints can have a big impact on the complexity of the problem. For instance, if only a few islands in the search space are feasible or a large number of constraints ($|J|+|K|$) need to be satisfied. For genetic algorithms satisfying equality constraints can be rather challenging. Thus, this needs to be addressed differently, for instance, by mapping the search space to a utility space where the equality constraints are always satisfied or injecting the knowledge of the equality constraint through customization.

+++

**Multi-modality.** Most aspects discussed so far are most likely known or to be relatively easy to define. However, the nature of the fitness landscape is less obvious bet yet essential to be aware of. In the case of multi-modal fitness landscapes, optimization becomes inevitably more difficult due to the existence of a few or even many local optima. For the solution found, one must always ask if the method has explored enough regions in the search space to maximize the probability of obtaining the global optimum. A multi-modal search space quickly shows the limitation of local search, which can easily get stuck.

+++

**Differentiability.** A function being differentiable implies the first or even second-order derivative can be calculated. Differentiable functions allow gradient-based optimization methods to be used, which can be a great advantage over gradient-free methods. The gradient provides a good indication of what direction shall be used for the search. Most gradient-based algorithms are point-by-point based and can be highly efficient for rather unimodal fitness landscapes.
However, in practice, often functions are non-differentiable, or a more complicated function requires a global instead of a local search. The research field addressing problems without knowing their mathematical optimization is also known as black-box optimization.

+++

**Evaluation Time.** Many optimization problems in practice consist of complicated and lengthy mathematical equations or domain-specific software to be evaluated. The usage of third-party software often results in a computationally expensive and time-consuming function for evaluating objectives or constraints. For those types of problems, the algorithm's overhead for determining the next solutions to be evaluated is often neglectable. A commercial software performing an evaluation often comes with various more practical issues such as distributed computing, several instances to be used in parallel and software license, and the software's possible failure for specific design variable combinations.

+++

**Uncertainty.** Often it is assumed that the objective and constraint functions are of a deterministic manner. However, if one or multiple target functions are nondeterministic, this introduces noise or also referred to as uncertainty. One technique to address the underlying randomness is to repeat the evaluation for different random seeds and average the resulting values. Moreover, the standard deviation derived from multiple evaluations can be utilized to determine the performance and the reliability of a specific solution. In general, optimization problems with underlying uncertainty are investigated by the research field called stochastic optimization. 

+++

Of course, this shall not be an exhaustive list of problem characteristics but rather an idea of how fundamentally different optimization problems are. Being aware of possible challenges, one can make better decisions regarding a method and its suitability. In this tutorial, solving a constrained bi-objective optimization problem is demonstrated. This, and your problem's characteristics, shall help you to use *pymoo* as a toolbox to tackle your optimization problem.
