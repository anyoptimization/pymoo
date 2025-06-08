# PyMoo Documentation Management Guide

This file contains important knowledge and best practices for managing the PyMoo documentation system.

## Documentation System Overview

PyMoo uses a hybrid documentation system with:
- **Markdown (.md) files**: Source files for documentation content
- **Jupyter Notebooks (.ipynb)**: Generated from .md files for interactive execution
- **Sphinx**: Documentation generator that processes both formats
- **Jupytext**: Tool that syncs between .md and .ipynb formats

## Key File Structure

```
docs/
├── run.sh                    # Script to convert .md to .ipynb and execute them
├── requirements.txt          # Python dependencies for documentation build
└── source/
    ├── conf.py              # Sphinx configuration
    ├── references.bib       # Bibliography for citations
    ├── algorithms/
    │   ├── index.md         # Algorithm section toctree
    │   ├── moo/             # Multi-objective algorithms
    │   └── soo/             # Single-objective algorithms
    ├── operators/
    │   └── index.md         # Operators section toctree
    ├── case_studies/
    │   └── index.md         # Case studies section toctree
    └── [other sections...]
```

## Essential Commands

### Converting Markdown to Notebooks
```bash
# Convert only missing notebooks (recommended)
docs/run.sh

# Force convert all notebooks (use sparingly)
docs/run.sh --force
```

### Managing Jupyter Dependencies
```bash
# Install jupytext if missing
pip install jupytext
```

## Critical Workflow Rules

### 1. File Editing Priorities
- ✅ **ALWAYS edit .md files** - these are the source of truth
- ❌ **NEVER edit .ipynb files directly** - they get overwritten
- 🔄 **Always regenerate .ipynb after .md changes** using `docs/run.sh`

### 2. When to Regenerate Notebooks
After editing any .md file:
```bash
# Remove the corresponding .ipynb file
rm docs/source/path/to/file.ipynb

# Regenerate it
docs/run.sh
```

### 3. Factory References (REMOVED)
The `pymoo.factory` module was removed. Replace any references:
```python
# OLD (causes errors)
from pymoo.factory import get_crossover

# NEW (correct)
from pymoo.core.crossover import Crossover
```

## Sphinx Configuration Fixes Applied

### 1. Updated docs/requirements.txt
```
sphinx==3.5.4
jinja2==2.10.1
markupsafe==1.1.1
sphinxcontrib-serializinghtml==1.1.4
nbsphinx==0.8.8
pydata-sphinx-theme==0.4.0
pandoc<3.0.0,>=1.12.1
```

### 2. Fixed conf.py Intersphinx URLs
```python
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None)
}
```

### 3. Fixed Numpydoc Format
In visualization files, changed:
```python
# BAD (causes warnings)
"""
Parameters
----------------
```

# GOOD (correct format)
"""
Parameters
----------
```

## Citation Management

### Adding Citations
1. Add BibTeX entry to `docs/source/references.bib`
2. Reference in text using: `<cite data-cite="citation_key"></cite>`
3. For manual citations, reference with: `[CitationKey]_`

### Example Citation Fix
```markdown
# In the text
The NRBO algorithm [Sowmya2024]_ is a population-based method...

# In references section
.. [Sowmya2024] Full citation text here.
```

## Table of Contents (Toctree) Management

### Adding New Documents to Navigation
For each new .md/.ipynb file, add to appropriate index.md:

```rst
.. toctree::
   :hidden:
   :maxdepth: 2

   existing/file
   new/file    # Add new entries here
```

### Common Index Files
- `algorithms/index.md` - For new algorithms
- `operators/index.md` - For new operators  
- `case_studies/index.md` - For new case studies
- `problems/index.md` - For new problems

## Run.sh Script Details

### Script Location and Usage
```bash
# Can be run from main workspace directory
/path/to/pymoo/docs/run.sh

# Excludes directories starting with underscore
# Processes only missing .ipynb files by default
```

### Script Configuration
The script automatically:
- Finds .md files using the script's location as reference
- Excludes `_static`, `_templates`, and other `_*` directories
- Converts .md to .ipynb using jupytext
- Executes notebooks to ensure they work

## Common Issues and Solutions

### 1. "Factory Module Not Found" Errors
- **Cause**: References to old `pymoo.factory` module
- **Fix**: Remove autofunction directives pointing to factory module
- **Files to check**: All operator .md files

### 2. "Unknown Target Name" Warnings
- **Cause**: Malformed HTML in Markdown tables
- **Fix**: Replace complex HTML with simple Markdown syntax
- **Common**: Tables with `<ul><br><li>` tags

### 3. "Document Not in Toctree" Warnings
- **Cause**: New files not added to navigation
- **Fix**: Add file path to appropriate index.md toctree

### 4. "Unreferenced Citations" Warnings
- **Cause**: Citations defined but not referenced in text
- **Fix**: Add proper citation reference in relevant text

### 5. Pandoc Version Conflicts
- **Cause**: Incompatible pandoc version
- **Fix**: Pin version in requirements.txt: `pandoc<3.0.0,>=1.12.1`

## Testing Changes

### Before Submitting Changes
1. Run `docs/run.sh` to ensure notebooks generate properly
2. Check for any Sphinx warnings in build output
3. Verify all new files are in appropriate toctrees
4. Ensure citations are properly referenced

### Expected Warnings (Safe to Ignore)
- Notebooks not in toctree (if intentionally excluded)
- Some unreferenced citations (for research completeness)

## Directory Structure for New Content

### Adding New Algorithms
```
algorithms/
├── moo/new_algorithm.md      # Multi-objective
├── soo/new_algorithm.md      # Single-objective
└── index.md                  # Add to toctree here
```

### Adding New Operators
```
operators/
├── new_operator.md
└── index.md                  # Add to toctree here
```

### Adding New Case Studies
```
case_studies/
├── new_study.md
└── index.md                  # Add to toctree here
```

## Best Practices Summary

1. **Always edit .md files, never .ipynb directly**
2. **Run docs/run.sh after any .md changes**
3. **Add new files to appropriate toctrees**
4. **Use proper citation formats**
5. **Test documentation builds before committing**
6. **Keep HTML simple in Markdown tables**
7. **Pin dependency versions for stability**
8. **Remove old factory module references**

## Quick Reference Commands

```bash
# Full workflow for editing documentation
vim docs/source/path/to/file.md           # Edit markdown
rm docs/source/path/to/file.ipynb         # Remove old notebook
docs/run.sh                               # Regenerate notebook
# Commit both .md and .ipynb files

# Emergency reset of specific notebook
rm docs/source/path/to/file.ipynb
docs/run.sh

# Check for broken references
grep -r "factory\.get_" docs/source/ --include="*.md"
grep -r "incremental_Lattice" docs/source/ --include="*.md"
```

This guide should help maintain consistency and avoid common pitfalls when working with the PyMoo documentation system.