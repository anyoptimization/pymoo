#!/bin/bash

# Set environment variable to disable debugger validation
export PYDEVD_DISABLE_FILE_VALIDATION=1

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/source"

# Check for --force flag
FORCE=false
if [[ "$1" == "--force" ]]; then
    FORCE=true
    echo "Force mode enabled - processing all files"
else
    echo "Processing only missing notebook files (use --force to process all)"
fi

echo "Processing markdown files..."

# Loop over each markdown file
for md_file in $(find "$SOURCE_DIR" -name '*.md');
do
    # Get the corresponding notebook file path
    nb_file="${md_file%.md}.ipynb"
    
    # Check if we should process this file
    should_process=false
    
    if [ "$FORCE" = true ]; then
        should_process=true
        echo "Processing $md_file (force mode)..."
    elif [ ! -f "$nb_file" ]; then
        should_process=true
        echo "Processing $md_file (notebook missing)..."
    else
        echo "Skipping $md_file (notebook exists, use --force to override)"
    fi
    
    # Process the file if needed
    if [ "$should_process" = true ]; then
        # Sync the markdown file to create/update the corresponding notebook
        jupytext --to ipynb --sync "$md_file" 2>/dev/null
        
        # Execute the notebook if it exists
        if [ -f "$nb_file" ]; then
            echo "Executing $nb_file..."
            jupyter nbconvert --execute --to notebook --inplace "$nb_file" --log-level=ERROR 2>/dev/null
        else
            echo "Warning: $nb_file not found after sync"
        fi
    fi
done

echo "Documentation build complete!"
