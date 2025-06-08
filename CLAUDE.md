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

## Key Configurations and Environments

- There are two anaconda environments to be used. `default` use this if not instructed otherwise.

## Documentation Conventions

- Use the numpy docstring format. If you see somewhere it has not been used correctly, correct it.

## Documentation Troubleshooting

- The documentation compile log file is located docs/compile.log. Use it to locate errors when compiling the documentation