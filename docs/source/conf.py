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
# from _theme import *  # Disabled custom theme imports

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
    'sphinx_copybutton',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

bibtex_bibfiles = ['references.bib']

# ===========================================================================
# HTML
# ===========================================================================

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_book_theme'
# html_theme_path = ['.']  # Disabled custom theme
html_logo = "_static/logo.svg"
html_favicon = '_static/favicon.ico'

# Sphinx Book Theme configuration
html_theme_options = {
    "repository_url": "https://github.com/anyoptimization/pymoo",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": False,
    "use_edit_page_button": True,
    "use_source_button": True,
    "show_navbar_depth": 1,
    "collapse_navigation": True,
    "navigation_depth": 1,
    # Remove all navbar customizations to test basic functionality
    # "navbar_start": ["navbar-logo"],
    # "navbar_center": [],
    # "navbar_end": [],
    "logo": {
        "text": "pymoo",
        "image_light": "_static/logo.svg",
    },
    "nosidebar": True,  # Disable the right sidebar
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add custom CSS
html_css_files = [
    'css/custom.css',
]

# Add custom JS
html_js_files = [
    'js/pymoo-sidebar.js',
]

# Copy specific files to the root of the build directory
html_extra_path = ['_static/llms.txt', '_static/robots.txt']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']



# =========================================================================================================
# sphinx.ext.intersphinx - Mappings to other projects
# =========================================================================================================

intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy', None),
                       'matplotlib': ('https://matplotlib.org/stable', None)}

# ===========================================================================
# nbsphinx
# ===========================================================================


# Exclude build directory and Jupyter backup files:
exclude_patterns = ['build', '**.ipynb_checkpoints']

# Check mode for fast testing (triggered by make check)
CHECK_MODE = os.environ.get('PYMOO_DOCS_CHECK_MODE', '0') == '1'

if CHECK_MODE:
    print("Fast check mode enabled - excluding all ipynb files for faster testing")
    exclude_patterns.append("**/*.ipynb")


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
nbsphinx_codecell_lexer = 'python'

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

# ===========================================================================
# sphinx-copybutton
# ===========================================================================

# Configure copy button behavior
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True
copybutton_copy_empty_lines = False

# ===========================================================================
# nbsphinx
# ===========================================================================

# Disable RequireJS to avoid conflicts with clipboard.min.js
nbsphinx_requirejs_path = ""

# ===========================================================================
# Custom Source File Mapping for Edit Button
# ===========================================================================

def setup(app):
    """
    Custom setup function to modify the source file mapping.
    This ensures that the "Edit Source" button links to .md files instead of .ipynb files.
    """
    def html_page_context(app, pagename, templatename, context, doctree):
        """
        Add JavaScript to fix edit button URLs to point to .md files.
        """
        # Add JavaScript that will fix the edit button URLs after page load
        fix_edit_urls_js = """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Find all edit and source buttons and fix their URLs
            const editButtons = document.querySelectorAll('a[href*="/edit/"], a[href*="/blob/"]');
            editButtons.forEach(button => {
                if (button.href.includes('.ipynb')) {
                    button.href = button.href.replace('.ipynb', '.md');
                }
            });
        });
        </script>
        """
        
        # Add the JavaScript to the page context
        context['fix_edit_urls_js'] = fix_edit_urls_js
    
    # Connect the function to the html-page-context event
    app.connect('html-page-context', html_page_context)
    
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }


