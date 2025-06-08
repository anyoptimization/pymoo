#!/bin/bash

# Clean up all .ipynb files in the documentation source directory
# Ignores folders starting with underscore (_)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/source"

# Check for arguments
FORCE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Clean up all .ipynb files in the documentation source directory."
            echo "Ignores folders starting with underscore (_)."
            echo ""
            echo "Options:"
            echo "  --force, -f   Actually delete files (dry run by default)"
            echo "  --help, -h    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0            # Dry run - show what would be deleted"
            echo "  $0 --force    # Actually delete all .ipynb files"
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Cleaning up .ipynb files in: $SOURCE_DIR"
echo "Ignoring folders starting with underscore (_)"

if [[ "$FORCE" == false ]]; then
    echo ""
    echo "DRY RUN MODE - No files will be deleted"
    echo "Use --force to actually delete files"
fi

echo ""

# Find all .ipynb files (excluding folders starting with _)
IPYNB_FILES=($(find "$SOURCE_DIR" -name '*.ipynb' -not -path "*/\\_*/*"))
total_files=${#IPYNB_FILES[@]}

if [[ $total_files -eq 0 ]]; then
    echo "No .ipynb files found to clean up."
    exit 0
fi

echo "Found $total_files .ipynb files to clean up:"
echo ""

current_index=0

# Process each file
for ipynb_file in "${IPYNB_FILES[@]}"; do
    ((current_index++))
    
    # Get relative path for cleaner display
    rel_path="${ipynb_file#$SOURCE_DIR/}"
    
    if [[ "$FORCE" == true ]]; then
        echo "[$current_index/$total_files] Deleting: $rel_path"
        rm "$ipynb_file"
        if [[ $? -eq 0 ]]; then
            echo "  ✓ Deleted successfully"
        else
            echo "  ✗ Failed to delete"
        fi
    else
        echo "[$current_index/$total_files] Would delete: $rel_path"
    fi
done

echo ""

if [[ "$FORCE" == true ]]; then
    echo "Cleanup complete! Deleted $total_files .ipynb files."
else
    echo "Dry run complete! Found $total_files .ipynb files that would be deleted."
    echo "Use --force to actually delete these files."
fi