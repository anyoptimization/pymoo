# Documentation Conversion Scripts

This directory contains scripts for converting markdown files to Jupyter notebooks using jupytext.

## Scripts

### `md_to_ipynb.py`

Converts a single markdown file to a synced Jupyter notebook.

**Usage:**
```bash
python scripts/md_to_ipynb.py <markdown_file>
```

**Example:**
```bash
python md_to_ipynb.py source/algorithms/soo/ga.md
```

### `convert_all_md.py`

Recursively finds all markdown files in a directory and converts them to synced Jupyter notebooks.

**Usage:**
```bash
python convert_all_md.py [--dry-run] [--source-dir <path>] [--exclude <patterns>]
```

**Options:**
- `--dry-run`: Show what would be converted without actually doing it
- `--source-dir`: Source directory to search (default: `source`)
- `--exclude`: Patterns to exclude from conversion (default: `README.md`)

**Examples:**
```bash
# Convert all markdown files in source/
python convert_all_md.py

# Dry run to see what would be converted
python convert_all_md.py --dry-run

# Convert files in a specific directory
python convert_all_md.py --source-dir source/algorithms

# Exclude specific files
python convert_all_md.py --exclude README.md index.md
```

## Requirements

Make sure you have jupytext installed:
```bash
pip install jupytext
```

## How it works

Both scripts use jupytext with the `--sync` option, which:
1. Creates a paired Jupyter notebook (.ipynb) file
2. Keeps the notebook synced with the markdown file
3. Allows editing either file and automatically updates the other

The created notebooks will be in MyST markdown format, which is compatible with Sphinx documentation.