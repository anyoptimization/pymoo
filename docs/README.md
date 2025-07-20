# PyMoo Documentation Builder

A standalone documentation building pipeline for pymoo that can be executed using `uvx` without requiring conda environments or manual dependency management.

## Quick Start

From the pymoo root directory, build the complete documentation:

```bash
# Build everything (clean, compile notebooks, build HTML)
uvx --from ./docs pymoo-docs all

# Or step by step:
uvx --from ./docs pymoo-docs clean     # Clean previous builds
uvx --from ./docs pymoo-docs compile   # Convert .md to .ipynb
uvx --from ./docs pymoo-docs build     # Build HTML with Sphinx
```

The built documentation will be available at `docs/build/html/index.html`.

## Prerequisites

- Python 3.9+ with `uvx` installed
- PyMoo source code (for API documentation)

Install uvx if you don't have it:
```bash
pip install uvx
```

**Note**: The first run will take longer as uvx downloads and installs all dependencies in an isolated environment. Subsequent runs will be much faster.

## Commands

### `uvx --from ./docs pymoo-docs clean`
Removes all generated files:
- `docs/build/` directory (HTML output, doctrees)
- Generated `.ipynb` files (keeps manually created notebooks)

### `uvx --from ./docs pymoo-docs compile`
Converts Markdown files to Jupyter notebooks:
- Finds all `.md` files with jupytext metadata in `docs/source/`
- Converts them to `.ipynb` format using jupytext
- Executes notebooks to generate outputs

Options:
- `--skip-existing`: Only convert files that don't have existing notebooks
- `--force`: Force regeneration of all notebooks (overwrite existing)
- Specific files: `uvx --from ./docs pymoo-docs compile algorithms/nsga2.md`

### `uvx --from ./docs pymoo-docs build`
Builds HTML documentation using Sphinx:
- Processes all `.rst`, `.md`, and `.ipynb` files in `docs/source/`
- Generates API documentation from pymoo source code
- Outputs to `docs/build/html/`
- Copies Markdown files to build directory

### `uvx --from ./docs pymoo-docs serve`
Serves the built documentation locally:
```bash
uvx --from ./docs pymoo-docs serve --port 8000
```
Opens a web server at `http://localhost:8000`

### `uvx --from ./docs pymoo-docs check`
Fast build for testing (excludes heavy computations):
- Sets `PYMOO_DOCS_CHECK_MODE=1` environment variable
- Builds documentation with minimal content for quick validation

### `uvx --from ./docs pymoo-docs all`
Complete build pipeline:
1. Clean all generated files
2. Compile all Markdown files to notebooks
3. Build HTML documentation

Options:
- `--skip-existing`: Skip notebooks that already exist
- `--force`: Force regeneration of all files

## Examples

```bash
# From pymoo root directory:

# Quick check build
uvx --from ./docs pymoo-docs check

# Full clean build
uvx --from ./docs pymoo-docs all --force

# Update specific algorithm documentation
uvx --from ./docs pymoo-docs compile algorithms/nsga2.md
uvx --from ./docs pymoo-docs build

# Serve and view documentation
uvx --from ./docs pymoo-docs serve
# Open http://localhost:8000 in browser

# Compile specific files
uvx --from ./docs pymoo-docs compile installation.md faq.md

# Compile only missing notebooks (incremental)
uvx --from ./docs pymoo-docs compile --skip-existing

# Force regenerate all notebooks
uvx --from ./docs pymoo-docs compile --force

# Full build with incremental compilation
uvx --from ./docs pymoo-docs all --skip-existing
```

## Traditional Installation

If you prefer to install dependencies manually:

```bash
cd docs
pip install -e .

# Then use the CLI directly
pymoo-docs all
```

## Development Setup

For development work on the documentation:

```bash
cd docs
pip install -e ".[dev]"
```

## Environment Variables

- `PYMOO_DOCS_CHECK_MODE=1` - Enable fast build mode (set automatically by `check` command)

## Dependencies

The documentation builder includes all necessary dependencies:

- **Sphinx ecosystem**: sphinx, nbsphinx, sphinx-copybutton, pydata-sphinx-theme
- **Jupyter**: jupyterlab, nbconvert, jupytext for notebook conversion
- **Scientific computing**: numpy, scipy, matplotlib, pandas, scikit-learn
- **Optimization**: autograd, optuna, moocore, modact
- **Visualization**: seaborn, plotly, bokeh, holoviews
- **Additional**: dill, torch, cython

## File Structure

```
docs/
├── pyproject.toml          # Project configuration and dependencies
├── pymoo_docs/            # CLI package
│   ├── __init__.py
│   └── cli.py             # Main CLI implementation
├── source/                # Documentation source files
├── build/                 # Generated documentation (created during build)
├── requirements.txt       # Legacy requirements (still used by conda env)
└── README.md             # This file
```

## Migration from Conda Environment

The new setup replaces the need for the `pymoo-docs-stable-310` conda environment. All dependencies are now managed through the `pyproject.toml` file and can be installed on-demand with `uvx`.

### Old workflow:
```bash
conda activate pymoo-docs-stable-310
make html
```

### New workflow:
```bash
uvx --from ./docs pymoo-docs all
```

## Troubleshooting

### Missing Dependencies
If you encounter missing dependencies, they can be added to the `dependencies` list in `pyproject.toml`.

### Notebook Execution Errors
The `compile` command executes notebooks during conversion. If a notebook fails to execute, the build will continue but may produce incomplete documentation.

### Port Already in Use
If port 8000 is busy when using `serve`, specify a different port:
```bash
uvx --from ./docs pymoo-docs serve --port 8080
```

## Contributing

To add new dependencies or modify the build process:

1. Edit `pyproject.toml` to add dependencies
2. Modify `pymoo_docs/cli.py` to change build logic
3. Test with `uvx --from ./docs pymoo-docs all`

The goal is to keep the documentation build process simple and self-contained, requiring no manual environment setup.