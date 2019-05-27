
.. meta::
   :description: pymoo - Multi-objective Optimization in Python
   :keywords: pymoo, optimization, multi-objective optimization, evolutionary optimization, nsga2, nsga3


.. raw:: html

    <div align="right" style="margin-bottom: -20px">
      <b>Latest Version:</b> pymoo==0.3.1</b>
    </div>



pymoo
==============================================================================


The framework is available on PyPi and can be installed with:

::

    pip install -U Cython>=0.29 numpy>=1.15 pymoo


Please note, that the dependencies used in the command above must be fulfilled before compiling some modules of *pymoo*
and can, therefore, not be ensured to be installed prior compilation with the setup script.
More details about the installation can be found :ref:`here <installation>`.


.. image:: resources/images/getting_started.svg
   :target: getting_started.html
   :width: 40%
   :alt: Getting Started
   :align: left



|vspace|

.. |vspace| raw:: latex

   \vspace{5mm}

If you intend to use *pymoo* for **any** profit-making purposes, please contact `Julian Blank <http://www.cse.msu.edu/~blankjul/>`_.

We are currently working on a paper.
Meanwhile, if you have used our framework for research purposes, please cite us with:

::

    @misc{pymoo,
        author = {Julian Blank and Kalyanmoy Deb},
        title = {pymoo - {Multi-objective Optimization in Python}},
        howpublished = {https://pymoo.org}
    }


Features
------------------------------------------------------------------------------

**Algorithms:** :ref:`Genetic Algorithm <nb_ga>`, :ref:`Differential Evolution <nb_de>`, :ref:`NSGA-II <nb_nsga2>`,
:ref:`R-NSGA-II <nb_rnsga2>`,
:ref:`NSGA-III <nb_nsga3>`, :ref:`U-NSGA-III <nb_unsga3>`, :ref:`R-NSGA-III <nb_rnsga3>`, :ref:`MOEA/D <nb_moead>`

**Test Problems:** :ref:`Overview <nb_problems>` (single-objective, multi-objective and many-objective)

**Performance Indicators:** :ref:`Hypervolume <nb_hv>`, :ref:`GD <nb_gd>`, :ref:`IGD <nb_igd>`, :ref:`R-Metric <nb_rmetric>`

**Non-Dominated Sorting:** :ref:`Naive<nb_nds_naive>`, :ref:`Fast<nb_nds_fast>`,
:ref:`Best Order<nb_nds_best>`

**Random Generators:** :ref:`Custom <nb_numpy>`, :ref:`Python <nb_rnd>`, :ref:`Numpy <nb_rnd>`

**Selection:** :ref:`Random <nb_selection_random>`, :ref:`Tournament Selection <nb_selection_tournament>`

**Sampling:** :ref:`Random <nb_sampling_random>`, :ref:`Latin Hypercube Sampling <nb_sampling_lhs>`

**Crossover:** :ref:`Simulated Binary Crossover<nb_crossover_sbx>`, :ref:`Uniform Crossover<nb_crossover_uniform>`,
:ref:`Half Uniform Crossover<nb_crossover_half_uniform>`, :ref:`Differential Crossover<nb_crossover_differential>`,
:ref:`Point Crossover<nb_crossover_point>`, :ref:`Exponential Crossover<nb_crossover_exponential>`

**Mutation:** :ref:`Polynomial Mutation <nb_mutation_pm>`, :ref:`Bitflip Mutation<nb_mutation_bitflip>`


.. **Decomposition:** :ref:`ASF <nb_asf>`, :ref:`Tchebichef <nb_thebi>`

**Visualization:** :ref:`Scatter <nb_scatter>`, :ref:`Parallel Coordinate Plot (PCP) <nb_pcp>`, :ref:`Heatmap <nb_heat>` , :ref:`Petal Width<nb_petal_width>`,
:ref:`Radar <nb_radar>`, :ref:`Radviz <nb_radviz>`, :ref:`Star Coordinate <nb_star>`


About
------------------------------------------------------------------------------

This framework is developed and maintained by `Julian Blank <http://www.cse.msu.edu/~blankjul/>`_ who is affiliated to the
`Computational Optimization and Innovation Laboratory (COIN) <https://www.coin-laboratory.com>`_ supervised
by `Kalyanmoy Deb <https://www.egr.msu.edu/people/profile/kdeb>`_ at the Michigan State University in
East Lansing, Michigan, USA.
Each algorithms is developed as close as possible to the proposed version to the
best of our knowledge. **NSGA-II** and **NSGA-III** have been develop collaboratively with one of the authors
and, therefore, we recommend using them for **official** benchmarks.



News
------------------------------------------------------------------------------
**April 10, 2019:** The framework has reached a new degree of professionality by improving the
software documentation regarding tutorial and API.


Content
------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2

   installation
   getting_started.ipynb
   algorithms/index
   problems/index.ipynb
   components/index
   tutorial/index
   api/index
   versions
   references
   contact
   license


Indices and tables
------------------------------------------------------------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

