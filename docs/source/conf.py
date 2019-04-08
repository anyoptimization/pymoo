import sys, os

needs_sphinx = '2.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
   # 'sphinx.ext.autosectionlabel',
    'sphinxcontrib.bibtex',
]

compile_ipynb = True
exclude_patterns = [
    '_build',
    '.ipynb_checkpoints'
]

if not compile_ipynb:
    exclude_patterns.append('**.ipynb')

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../../pymoo'))

templates_path = ['templates']
source_suffix = '.rst'
master_doc = 'index'
project = u'pymoo'
copyright = u'2019, Julian Blank, Computational Optimization and Innovation Laboratory (COIN), Michigan State University'
version = '0.2.5'
release = '0.2.5'
exclude_patterns = ['build']
pygments_style = 'sphinx'

# -- Options for HTML output ---------------------------------------------------
html_theme = 'scipy'
html_theme_path = ['_theme']
html_static_path = ['_static']


html_theme_options = {
    "edit_link": "false",
    "sidebar": "right",
    "scipy_org_logo": "true",
    "rootlinks": [
        ("http://pymoo.org/", "pymoo.org"),
        ("https://github.com/msu-coinlab/pymoo", "GitHub"),
        ("http://pymoo.org/api.html", "API")]
}
#html_sidebars = {'index': ['indexsidebar.html', 'searchbox.html']}
