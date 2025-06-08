#!/bin/bash

# Set environment variable to disable debugger validation
export PYDEVD_DISABLE_FILE_VALIDATION=1

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/source"
PYMOO_ROOT="$(dirname "$SCRIPT_DIR")"

# Add pymoo to Python path
export PYTHONPATH="$PYMOO_ROOT:$PYTHONPATH"

# Set conda environment (default to "default" if not specified)
CONDA_ENV="${CONDA_ENV:-default}"
echo "Using conda environment: $CONDA_ENV"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Check for arguments
FORCE=false
EXECUTE=true
SPECIFIC_FILES=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --no-execute)
            EXECUTE=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options] [file1.md] [file2.md] ..."
            echo ""
            echo "Options:"
            echo "  --force       Process all files, not just missing notebooks"
            echo "  --no-execute  Convert markdown to notebooks but don't execute them"
            echo "  --usage       Show detailed usage examples"
            echo "  --help, -h    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Process missing notebooks only"
            echo "  $0 --force                           # Process all files"
            echo "  $0 file1.md file2.md                # Process specific files"
            echo "  $0 --no-execute file.md             # Convert but don't execute"
            echo ""
            echo "Use --usage for more detailed examples."
            exit 0
            ;;
        --usage)
            echo "DETAILED USAGE EXAMPLES"
            echo "======================="
            echo ""
            echo "Basic Usage:"
            echo "  $0"
            echo "    → Processes only files where the .ipynb is missing"
            echo "    → Executes all generated notebooks"
            echo ""
            echo "  $0 --force"
            echo "    → Processes ALL markdown files in docs/source/"
            echo "    → Regenerates existing notebooks"
            echo "    → Executes all notebooks"
            echo ""
            echo "Processing Specific Files:"
            echo "  $0 docs/source/algorithms/moo/nsga2.md"
            echo "    → Processes only nsga2.md if notebook missing"
            echo "    → Executes the notebook"
            echo ""
            echo "  $0 file1.md file2.md file3.md"
            echo "    → Processes multiple specific files"
            echo "    → Only if their notebooks are missing"
            echo "    → Executes generated notebooks"
            echo ""
            echo "  $0 --force file1.md file2.md"
            echo "    → Forces processing of specific files"
            echo "    → Regenerates notebooks even if they exist"
            echo "    → Executes all notebooks"
            echo ""
            echo "Skip Execution:"
            echo "  $0 --no-execute"
            echo "    → Converts missing .md to .ipynb but doesn't execute"
            echo "    → Useful for syntax checking or quick conversion"
            echo ""
            echo "  $0 --no-execute file1.md file2.md"
            echo "    → Converts specific files but doesn't execute"
            echo "    → Good for testing conversion without running code"
            echo ""
            echo "  $0 --force --no-execute"
            echo "    → Forces conversion of all files but skips execution"
            echo "    → Useful for bulk regeneration without time-consuming execution"
            echo ""
            echo "Combined Options:"
            echo "  $0 --force --no-execute docs/source/parallelization/custom.md"
            echo "    → Forces regeneration of specific file without execution"
            echo "    → Useful when you know the file exists but want to update it"
            echo ""
            echo "Directory Processing:"
            echo "  cd docs/source/algorithms/moo && $0 *.md"
            echo "    → Process all .md files in current directory"
            echo "    → Respects force and execution flags"
            echo ""
            echo "Common Workflows:"
            echo "  # After editing a markdown file:"
            echo "  $0 docs/source/path/to/edited-file.md"
            echo ""
            echo "  # Bulk regeneration after major changes:"
            echo "  $0 --force"
            echo ""
            echo "  # Quick syntax check without execution:"
            echo "  $0 --no-execute docs/source/path/to/file.md"
            echo ""
            echo "  # Update all files in a section:"
            echo "  $0 --force docs/source/algorithms/moo/*.md"
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [[ -f "$1" ]]; then
                SPECIFIC_FILES+=("$1")
            else
                echo "Error: File '$1' not found"
                exit 1
            fi
            shift
            ;;
    esac
done

# Display processing mode
if [[ ${#SPECIFIC_FILES[@]} -gt 0 ]]; then
    echo "Processing specific files: ${SPECIFIC_FILES[*]}"
elif [[ "$FORCE" == true ]]; then
    echo "Force mode enabled - processing all files"
else
    echo "Processing only missing notebook files (use --force to process all, or specify file paths)"
fi

if [[ "$EXECUTE" == true ]]; then
    echo "Notebook execution: enabled"
else
    echo "Notebook execution: disabled"
fi

echo "Processing markdown files..."

# Function to process a single markdown file
process_file() {
    local md_file="$1"
    local nb_file="${md_file%.md}.ipynb"
    
    echo "Processing $md_file..."
    
    # Sync the markdown file to create/update the corresponding notebook
    jupytext --to ipynb --sync "$md_file" 2>/dev/null
    
    # Execute the notebook if it exists and execution is enabled
    if [ -f "$nb_file" ]; then
        if [[ "$EXECUTE" == true ]]; then
            echo "Executing $nb_file..."
            jupyter nbconvert --execute --to notebook --inplace "$nb_file" --log-level=ERROR 2>/dev/null
        else
            echo "Skipping execution of $nb_file (--no-execute specified)"
        fi
    else
        echo "Warning: $nb_file not found after sync"
    fi
}

# Process files based on mode
if [[ ${#SPECIFIC_FILES[@]} -gt 0 ]]; then
    # Process specific files with force logic if applicable
    for md_file in "${SPECIFIC_FILES[@]}"; do
        nb_file="${md_file%.md}.ipynb"
        
        # Check if we should process this file (apply force logic for specific files too)
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
            process_file "$md_file"
        fi
    done
else
    # Loop over each markdown file (excluding folders starting with _)
    for md_file in $(find "$SOURCE_DIR" -name '*.md' -not -path "*/\\_*/*");
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
            process_file "$md_file"
        fi
    done
fi

echo "Documentation build complete!"
