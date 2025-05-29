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

.. _nb_de:
```

.. meta::
   :description: Differential Evolution (DE) is a genetic algorithm that uses the differentials between individuals to create the offspring population. Through the usage of differential, the recombination is rotation-invariant and self-adaptive.

+++

.. meta::
   :keywords: Differential Evolution, DE,  Multi-modal Optimization, Nature-inspired Algorithm, Single-objective Optimization, Python

+++

# DE: Differential Evolution

+++

The classical single-objective
differential evolution algorithm <cite data-cite="de"></cite> is where different crossover variations
and methods can be defined. It is known for its good results for
global optimization.

The differential evolution crossover is simply defined by:

$$
v = x_{\pi_1} + F \cdot (x_{\pi_2} - x_{\pi_3})
$$

where $\pi$ is a random permutation with 3 entries. The difference is taken between individual 2 and 3 and added to the first one. This is shown below:


<div style="text-align: center;">
    <img src="https://github.com/anyoptimization/pymoo-data/blob/main/docs/images/de_crossover.png?raw=true" width="350">
</div>




Then, a second crossover between an individual and the so-called donor vector $v$ is performed. The second crossover can be simply binomial/uniform or exponential.



A great tutorial and more detailed information can be found [here](https://web.archive.org/web/20190928024126/http://www1.icsi.berkeley.edu/~storn/code.html). The following guideline is copied from the description there (variable names are modified):

If you are going to optimize your own objective function with DE, you may try the following classical settings for the input file first: Choose method e.g. DE/rand/1/bin, set the population size to 10 times the number of parameters, select weighting factor F=0.8, and crossover constant CR=0.9. 
Recently, it has been found that selecting F from the interval (0.5, 1.0) randomly for each generation or each difference vector, a technique called dither, improves convergence behavior significantly, especially for noisy objective functions. 


It has also been found that setting CR to a low value, e.g., `CR=0.2` helps to optimize separable functions since it fosters the search along the coordinate axes. On the contrary, this choice is not effective if parameter dependence is encountered, which frequently occurs in real-world optimization problems rather than artificial test functions. So for parameter dependence, the choice of `CR=0.9` is more appropriate. Another interesting empirical finding is that raising NP above, say, 40 does not substantially improve the convergence, independent of the number of parameters. It is worthwhile to experiment with these suggestions. Ensure that you initialize your parameter vectors by exploiting their full numerical range, i.e., if a parameter is allowed to exhibit values in the range (-100, 100), it is a good idea to pick the initial values from this range instead of unnecessarily restricting diversity.


Keep in mind that different problems often require different settings for NP, F, and CR (have a look into the different papers to get a feeling for the settings). If you still get misconvergence, you might want to try a different method. We mostly use 'DE/rand/1/...' or 'DE/best/1/...'. The crossover method is not so crucial, although Ken Price claims that binomial is never worse than exponential. In the case of misconvergence, also check your choice of objective function. There might be a better one to describe your problem. Any knowledge that you have about the problem should be worked into the objective function. A good objective function can make all the difference.

And this is how DE can be used:

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.problems import get_problem
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize


problem = get_problem("ackley", n_var=10)


algorithm = DE(
    pop_size=100,
    sampling=LHS(),
    variant="DE/rand/1/bin",
    CR=0.3,
    dither="vector",
    jitter=False
)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.soo.nonconvex.de.DE
    :noindex:
```
