




.. meta::
   :description: pymoo - Multi-objective Optimization in Python
   :keywords: pymoo, optimization, multi-objective optimization, evolutionary optimization, nsga2, nsga3



.. |blankjul| raw:: html

   <a href="http://www.cse.msu.edu/~blankjul/" target="_blank">Julian Blank</a>

.. |kdeb| raw:: html

   <a href="https://www.egr.msu.edu/people/profile/kdeb" target="_blank">Kalyanmoy Deb</a>

.. |github| raw:: html

   <a href="https://github.com/msu-coinlab/pymoo" target="_blank">GitHub</a>

.. |issues| raw:: html

   <a href="https://github.com/msu-coinlab/pymoo/issues" target="_blank">Issue Tracker</a>


.. |coin| raw:: html

   <a href="https://www.coin-laboratory.com" target="_blank">Computational Optimization and Innovation Laboratory (COIN)</a>


Our framework offers state of the art single- and multi-objective algorithms and many
more features related to multi-objective optimization such as visualization and decision making.
**pymoo** is available on PyPi and can be installed by:

::

    pip install -U pymoo


Please note, that some modules can be compiled to speed up computations (optional). By using the command
above, an attempt is made to compile the modules, however, if unsuccessful the
plain python version is installed. More information are available in our 
:ref:`Installation Guide <installation>`.

To get familiar with our framework we recommended having a look at our
getting started guide:

.. image:: resources/images/getting_started.svg
   :target: getting_started.html
   :width: 40%
   :alt: Getting Started
   :align: left



Features
------------------------------------------------------------------------------

Furthermore, our frameworks offers a variety of different features which cover various facets of multi-objective optimization:

.. include:: portfolio.rst



Reference
------------------------------------------------------------------------------

We are currently working on a paper.
Meanwhile, if you have used our framework for research purposes, you can cite us with:

::

    @misc{pymoo,
        author = {Julian Blank and Kalyanmoy Deb},
        title = {pymoo - {Multi-objective Optimization in Python}},
        howpublished = {https://pymoo.org}
    }



News
------------------------------------------------------------------------------
.. include:: news_current.rst
:ref:`More News<news>`


About
------------------------------------------------------------------------------

This framework is developed and maintained by |blankjul| who is affiliated to the
|coin| supervised
by |kdeb| at the Michigan State University in
East Lansing, Michigan, USA.

We have developed the framework for research purposes and hope to contribute to the research area by delivering tools
for solving and analyzing multi-objective problems. Each algorithms is developed as close as possible to the proposed
version to the best of our knowledge.
**NSGA-II** and **NSGA-III** have been developed collaboratively with one of the authors and, therefore, we recommend
using them for **official** benchmarks.

If you intend to use our framework for **any** profit-making purposes, please contact us. Also, be aware that even
state of the art algorithms are just the starting point for many optimization problems.
The full potential of genetic algorithms requires customization and the incorporation of domain knowledge.
We have more than 20 years experience in the optimization field and are eager to tackle challenging problems.
Let us know if you are interested in working with experienced collaborators in optimization. Please keep in mind
that only through such projects we are able to keep developing and improving our framework and make sure
it meets the current needs of the industry.

Moreover, any kind of **contribution** is more than welcome:

.. |star| image:: resources/images/star.png
  :height: 25
  :target: https://github.com/msu-coinlab/pymoo

.. raw:: html

  <div style="margin-left: 10px;">

**(i)** Give us a |star| on |github|.
This makes not only our framework but in general multi-objective optimization more 
popular by being listed with a higher rank regarding specific keywords.

**(ii)** In order to offer more and more new algorithms and features, we are more than 
happy if somebody wants to contribute by developing code. You can see it as a 
win-win situations, because your development will be linked to your publication(s) which
can significantly increase the awareness of your work. Please note that, we aim to keep 
a high-level of code quality and some refactorings might be suggested. We have prepared
a list of `suggested contributions <contributions.html>`_.


**(iii)** You like our framework and you would like to use it for profit-making purposes?
We are always searching for industrial collaborations because they help to direct research to meet 
the needs the of the industry. In our laboratory solving practical problems has a high priority 
for every student and can help you to benefit from our research experience we have gained
over the last years.

.. raw:: html

  </div>



If you find a bug or you have any kind of concern regarding correctness please use 
our |issues| Nobody is perfect
and only if we are aware of issues we can start to investigate them.



Content
------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2

   news
   installation
   getting_started.ipynb
   algorithms/index.ipynb
   problems/index.ipynb
   operators/index.ipynb
   visualization/index.ipynb
   decision_making/index.ipynb
   misc/performance_indicator.ipynb
   misc/index.ipynb
   tutorial/index
   api/index
   versions
   contributions.ipynb
   references
   contact
   license




Indices and tables
------------------------------------------------------------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


