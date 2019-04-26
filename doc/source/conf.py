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
# sphinx
# =========================================================================================================


templates_path = ['templates']
source_suffix = '.rst'
master_doc = 'index'
project = u'pymoo'
copyright = u'2019, Julian Blank, Michigan State University'
version = '0.3.0'
release = '0.3.0'
pygments_style = 'sphinx'

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

# Exclude build directory and Jupyter backup files:
exclude_patterns = ['build', '**.ipynb_checkpoints']

# Default language for syntax highlighting in reST and Markdown cells
highlight_language = 'none'

# Don't add .txt suffix to source files (available for Sphinx >= 1.5):
html_sourcelink_suffix = ''

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


# =========================================================================================================
# Numpydoc
# ===========================================================================


# Whether to produce plot:: directives for Examples sections that contain import matplotlib or from matplotlib import.
numpydoc_use_plots = True

# Whether to show all members of a class in the Methods and Attributes sections automatically. True by default.
numpydoc_show_class_members = True

# Whether to show all inherited members of a class in the Methods and Attributes sections automatically.
# If it’s false, inherited members won’t shown. True by default.
numpydoc_show_inherited_class_members = False

# Whether to create a Sphinx table of contents for the lists of class methods and attributes.
# If a table of contents is made, Sphinx expects each entry to have a separate page. True by default.
numpydoc_class_members_toctree = False

# A regular expression matching citations which should be mangled to avoid conflicts due to duplication across
# the documentation. Defaults to [\w-]+.
#numpydoc_citation_re = False

# Until version 0.8, parameter definitions were shown as blockquotes, rather than in a definition list.
# If your styling requires blockquotes, switch this config option to True. This option will be removed in version 0.10.
numpydoc_use_blockquotes = False

# Deprecated since version edit: your HTML template instead. Whether to insert an edit link after docstrings.
numpydoc_edit_link = False

# =========================================================================================================
# autodoc - import the library
# =========================================================================================================

sys.path.insert(0, os.path.abspath('../../../pymoo'))

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
    "rootlinks": links_remote
}

# html_sidebars = {'index': ['indexsidebar.html', 'searchbox.html']}
