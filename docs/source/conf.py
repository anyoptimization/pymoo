# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
from os.path import dirname

SOURCE = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, SOURCE)
from _theme import *

ROOT = dirname(dirname(SOURCE))
sys.path.insert(0, ROOT)

import pymoo
from pymoo.config import Config
Config.parse_custom_docs = True

DEBUG = True

# -- Project information -----------------------------------------------------

project = 'pymoo: Multi-objective Optimization in Python'
copyright = '2020'
author = 'Julian Blank'

version = pymoo.__version__
release = version


# ===========================================================================
# General
# ===========================================================================


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
    'nbsphinx',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'matplotlib.sphinxext.plot_directive',
    'sphinxcontrib.bibtex',
]

bibtex_bibfiles = ['references.bib']

# ===========================================================================
# HTML
# ===========================================================================

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = '_theme'
html_theme_path = ['.']
html_logo = "_static/logo.svg"
html_favicon = '_static/favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']



# =========================================================================================================
# sphinx.ext.intersphinx - Mappings to other projects
# =========================================================================================================

intersphinx_mapping = {'python': ('http://docs.python.org/2', None),
                       'numpy': ('http://docs.scipy.org/doc/numpy', "http://docs.scipy.org/doc/numpy/objects.inv"),
                       'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
                       'matplotlib': ('http://matplotlib.sourceforge.net', None)}

# ===========================================================================
# nbsphinx
# ===========================================================================


# Exclude build directory and Jupyter backup files:
exclude_patterns = ['build', '**.ipynb_checkpoints']
if DEBUG:
    # exclude_patterns.append("**ipynb")


    # exclude_patterns.append("getting_started*")
    # exclude_patterns.append("interface*")
    # exclude_patterns.append("problems*")
    #
    # exclude_patterns.append("problems/single/*")
    # exclude_patterns.append("problems/multi/*")
    # exclude_patterns.append("problems/many/*")
    # exclude_patterns.append("problems/constrained/*")
    #
    # exclude_patterns.append("algorithms*")
    # exclude_patterns.append("customization*")
    # exclude_patterns.append("operators*")
    # exclude_patterns.append("visualization*")
    # exclude_patterns.append("api*")
    # exclude_patterns.append("decision_making*")
    # exclude_patterns.append("misc*")

    pass


# Default language for syntax highlighting in reST and Markdown cells
highlight_language = 'none'

# Don't add .txt suffix to source files (available for Sphinx >= 1.5):
html_sourcelink_suffix = ''

autoclass_content = 'init'

# Work-around until https://github.com/sphinx-doc/sphinx/issues/4229 is solved:
html_scaled_image_link = False

# Execute notebooks before conversion: 'always', 'never', 'auto' (default)
nbsphinx_execute = 'never'

# Use this kernel instead of the one stored in the notebook metadata:
# nbsphinx_kernel_name = 'python3'

# List of arguments to be passed to the kernel that executes the notebooks:
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# If True, the build process is continued even if an exception occurs:
# nbsphinx_allow_errors = True

# Controls when a cell will time out (defaults to 30; use -1 for no timeout):
# nbsphinx_timeout = 60

# Default Pygments lexer for syntax highlighting in code cells:
# nbsphinx_codecell_lexer = 'ipython3'

# Width of input/output prompts used in CSS:
# nbsphinx_prompt_width = '8ex'

# If window is narrower than this, input/output prompts are on separate lines:
# nbsphinx_responsive_width = '700px'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'friendly'


# ===========================================================================
# Numpydoc
# ===========================================================================


# Whether to produce plot:: directives for Examples sections that contain import matplotlib or from matplotlib import.
numpydoc_use_plots = True

# Whether to show all members of a class in the Methods and Attributes sections automatically. True by default.
numpydoc_show_class_members = True

# Whether to show all inherited members of a class in the Methods and Attributes sections automatically.
# If it’s false, inherited members won’t shown. True by default.
numpydoc_show_inherited_class_members = False

# Whether to create a Sphinx table of contents for the lists of class algorithms and attributes.
# If a table of contents is made, Sphinx expects each entry to have a separate page. True by default.
numpydoc_class_members_toctree = False

# A regular expression matching citations which should be mangled to avoid conflicts due to duplication across
# the documentation. Defaults to [\w-]+.
# numpydoc_citation_re = False

# Until version 0.8, parameter definitions were shown as blockquotes, rather than in a definition list.
# If your styling requires blockquotes, switch this config option to True. This option will be removed in version 0.10.
numpydoc_use_blockquotes = False

# Deprecated since version edit: your HTML template instead. Whether to insert an edit link after docstrings.
numpydoc_edit_link = False

