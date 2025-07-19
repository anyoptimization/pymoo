.. meta::
   :description: pymoo: An open source framework for multi-objective optimization in Python.
          It provides not only state of the art single- and multi-objective optimization algorithms but also many
          more features related to multi-objective optimization such as visualization and decision making.
   :keywords: Multi-objective Optimization, Evolutionary Algorithm, NSGA2


.. |anyopt| raw:: html

   <a href="https://anyoptimization.com" target="_blank">anyoptimization</a>


.. |blankjul| raw:: html

   <a href="http://julianblank.com" target="_blank">Julian Blank</a>

.. |kdeb| raw:: html

   <a href="https://www.egr.msu.edu/people/profile/kdeb" target="_blank">Kalyanmoy Deb</a>

.. |github| raw:: html

   <a href="https://github.com/anyoptimization/pymoo" target="_blank">GitHub</a>

.. |issues| raw:: html

   <a href="https://github.com/anyoptimization/pymoo/issues" target="_blank">Issue Tracker</a>


.. |coin| raw:: html

   <a href="http://www.coin-lab.org" target="_blank">Computational Optimization and Innovation Laboratory (COIN)</a>




pymoo: Multi-objective Optimization in Python
------------------------------------------------------------------------------


Our framework offers state of the art single- and multi-objective optimization algorithms and many
more features related to multi-objective optimization such as visualization and decision making.
**pymoo** is available on PyPi and can be installed by:

::

    pip install -U pymoo


Please note that some modules can be compiled to speed up computations (optional). The command
above attempts is made to compile the modules; however, if unsuccessful, the
pure python version is installed. More information are available in our 
:ref:`Installation Guide <installation>`.


.. raw:: html
   :file: home/cards.html




Features
********************************************************************************


Furthermore, our framework offers a variety of different features which cover various facets of multi-objective optimization:

.. include:: home/portfolio.rst



List Of Algorithms
********************************************************************************


.. csv-table:: Algorithms available in pymoo
   :header: "Algorithm", "Class", "Convenience", "Objective(s)", "Constraints", "Description"
   :widths: 60, 10, 10, 10, 10, 100
   :file: algorithms/algorithms.csv





Cite Us
********************************************************************************

If you have used our framework for research purposes, you can cite our publication by:

| `J. Blank and K. Deb, pymoo: Multi-Objective Optimization in Python, in IEEE Access, vol. 8, pp. 89497-89509, 2020, doi: 10.1109/ACCESS.2020.2990567 <https://ieeexplore.ieee.org/document/9078759>`_
|
| BibTex:

::

    @ARTICLE{pymoo,
        author={J. {Blank} and K. {Deb}},
        journal={IEEE Access},
        title={pymoo: Multi-Objective Optimization in Python},
        year={2020},
        volume={8},
        number={},
        pages={89497-89509},
    }


News
********************************************************************************
.. include:: news_current.rst
:ref:`More News<news>`


About
********************************************************************************

This framework is powered by |anyopt|, a Python research community. It is developed and maintained by |blankjul| who is affiliated to the
|coin| supervised
by |kdeb| at the Michigan State University in
East Lansing, Michigan, USA.

We have developed the framework for research purposes and hope to contribute to the research area by delivering tools
for solving and analyzing multi-objective problems. Each algorithm is developed as close as possible to the proposed
version to the best of our knowledge.
**NSGA-II** and **NSGA-III** have been developed collaboratively with one of the authors and, therefore, we recommend
using them for **official** benchmarks.

If you intend to use our framework for **any** profit-making purposes, please contact us. Also, be aware that even
state-of-the-art algorithms are just the starting point for many optimization problems.
The full potential of genetic algorithms requires customization and the incorporation of domain knowledge.
We have experience for more than 20 years in the optimization field and are eager to tackle challenging problems.
Let us know if you are interested in working with experienced collaborators in optimization. Please keep in mind
that only through such projects can we keep developing and improving our framework and making sure
it meets the industry's current needs.

Moreover, any kind of **contribution** is more than welcome:

.. |star| image:: _static/star.png
  :height: 25
  :target: https://github.com/anyoptimization/pymoo

.. raw:: html

  <div style="margin-left: 10px;">


**(i)** Give us a |star| on |github|.
This makes not only our framework but, in general, multi-objective optimization more accessible by being listed with a higher rank regarding specific keywords.

**(ii)** To offer more and more new algorithms and features, we are more than happy if somebody wants to contribute by developing code. You can see it as a
win-win situation because your development will be linked to your publication(s), which
can significantly increase your work awareness. Please note that we aim to keep a high level of code quality, and some refactoring might be suggested.


**(iii)** You like our framework, and you would like to use it for profit-making purposes?
We are always searching for industrial collaborations because they help direct research to meet the industry's needs. Our laboratory solving practical problems have a high priority for every student and can help you benefit from the research experience we have gained
over the last years.


.. raw:: html

  </div>



If you find a bug or you have any kind of concern regarding the correctness, please use
our |issues| Nobody is perfect
Moreover, only if we are aware of the issues we can start to investigate them.




Content
********************************************************************************

.. toctree::
   :maxdepth: 2

   news
   installation.ipynb
   getting_started/index.ipynb
   interface/index.ipynb
   problems/index.ipynb
   parallelization/index.ipynb
   algorithms/index.ipynb
   constraints/index.ipynb
   gradients/index.ipynb
   customization/index.ipynb
   operators/index.ipynb
   visualization/index.ipynb
   mcdm/index.ipynb
   case_studies/index.ipynb
   misc/indicators.ipynb
   misc/index.ipynb
   faq.ipynb
   api/index
   versions.ipynb
   contribute.ipynb
   references
   contact
   license





Contact
********************************************************************************

| `Julian Blank <http://julianblank.com>`_  
| (blankjul [at] outlook.com)

