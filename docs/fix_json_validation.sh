#!/bin/bash

# Script to fix JSON validation errors in notebooks
# Usage: ./fix_json_validation.sh [file1.md] [file2.md] ...
# If no files specified, processes all .md files with corresponding .ipynb files

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/source"

echo "Fixing JSON validation errors in notebooks..."

if [[ $# -gt 0 ]]; then
    # Process specific files
    for md_file in "$@"; do
        if [[ -f "$md_file" ]]; then
            nb_file="${md_file%.md}.ipynb"
            echo "Regenerating $nb_file from $md_file..."
            rm -f "$nb_file"
            jupytext --to ipynb "$md_file"
        else
            echo "Warning: $md_file not found"
        fi
    done
else
    # Process all .md files that have corresponding .ipynb files
    echo "Scanning for .md files with corresponding .ipynb files..."
    
    find "$SOURCE_DIR" -name "*.md" -not -path "*/\\_*/*" | while read md_file; do
        nb_file="${md_file%.md}.ipynb"
        if [[ -f "$nb_file" ]]; then
            echo "Regenerating $nb_file from $md_file..."
            rm -f "$nb_file"
            jupytext --to ipynb "$md_file"
        fi
    done
fi

echo "JSON validation fix complete!"