import sys, os

# sphinx version what should be used to compiling the documentation
needs_sphinx = '2.0'


# =========================================================================================================
# Extensions
# =========================================================================================================


extensions = [

    # for creating the APi
    'sphinx.ext.autodoc',

    # API use the auto summaries
    'sphinx.ext.autosummary',

    # type of comments used for docstrings
    'numpydoc',

    # easy use of jupyter notebooks
    'nbsphinx',

    # enables to provide links alias in the project
    'sphinx.ext.intersphinx',

    'sphinx.ext.coverage',

    'matplotlib.sphinxext.plot_directive',

    # for the reference page and citing
    'sphinxcontrib.bibtex',

]


# =========================================================================================================
# sphinx.ext.intersphinx - Mappings to other projects
# =========================================================================================================

intersphinx_mapping = {'python': ('http://docs.python.org/2', None),
                       'numpy': ('http://docs.scipy.org/doc/numpy', "http://docs.scipy.org/doc/numpy/objects.inv"),
                       'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
                       'matplotlib': ('http://matplotlib.sourceforge.net', None)}


# =========================================================================================================
# nbsphinx - Using jupyter notebooks in this documentation
# =========================================================================================================

compile_ipynb = True
exclude_patterns = [
    '_build',
    '.ipynb_checkpoints'
]

if not compile_ipynb:
    exclude_patterns.append('**.ipynb')

# =========================================================================================================
# autodoc - import the library
# =========================================================================================================

sys.path.insert(0, os.path.abspath('../../../pymoo'))


# =========================================================================================================
# sphinx
# =========================================================================================================


templates_path = ['templates']
source_suffix = '.rst'
master_doc = 'index'
project = u'pymoo'
copyright = u'2019, Julian Blank, Computational Optimization and Innovation Laboratory (COIN), Michigan State University'
version = '0.2.5'
release = '0.2.5'
exclude_patterns = ['build']
pygments_style = 'sphinx'


# =========================================================================================================
# html
# =========================================================================================================

html_theme = 'scipy'
html_theme_path = ['_theme']
html_static_path = ['_static']


links_local = [
        ("http://localhost:8001", "pymoo.org"),
        ("https://github.com/msu-coinlab/pymoo", "GitHub"),
        ("http://localhost:8001/api/index.html", "API")
]

links_remote = [
        ("http://pymoo.org/", "pymoo.org"),
        ("https://github.com/msu-coinlab/pymoo", "GitHub"),
        ("http://pymoo.org/api/index.html", "API")
]

html_theme_options = {
    "edit_link": "false",
    "sidebar": "right",
    "scipy_org_logo": "true",
    "rootlinks": links_local
}
#html_sidebars = {'index': ['indexsidebar.html', 'searchbox.html']}
