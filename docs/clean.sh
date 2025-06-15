#!/bin/bash

# Clean up generated documentation files in the documentation source directory
# Removes .ipynb files, .ipynb_checkpoints directories, and build artifacts
# Ignores folders starting with underscore (_)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/source"
BUILD_DIR="$SCRIPT_DIR/build"

# Check for arguments
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry)
            DRY_RUN=true
            shift
            ;;
        --force|-f)
            DRY_RUN=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Clean up generated documentation files in the documentation source directory."
            echo "Removes .ipynb files, .ipynb_checkpoints directories, and build artifacts."
            echo "Ignores folders starting with underscore (_)."
            echo ""
            echo "Options:"
            echo "  --dry         Dry run - show what would be deleted"
            echo "  --force, -f   Same as default (actually delete files)"
            echo "  --help, -h    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0            # Actually delete all generated files"
            echo "  $0 --dry      # Dry run - show what would be deleted"
            echo "  $0 --force    # Actually delete all generated files (same as default)"
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Cleaning generated documentation files in: $SOURCE_DIR"
echo "Ignoring folders starting with underscore (_)"

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "DRY RUN MODE - No files will be deleted"
    echo "Run without --dry to actually delete files"
fi

echo ""

# Function to safely remove files/directories
safe_remove() {
    local path="$1"
    local type="$2"
    local rel_path="${path#$SOURCE_DIR/}"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "Would delete $type: $rel_path"
    else
        echo "Deleting $type: $rel_path"
        if [[ "$type" == "directory" ]]; then
            rm -rf "$path"
        else
            rm "$path"
        fi
        
        if [[ $? -eq 0 ]]; then
            echo "  ✓ Deleted successfully"
        else
            echo "  ✗ Failed to delete"
        fi
    fi
}

# Find and clean different types of generated files
total_cleaned=0

# 1. Clean .ipynb files (excluding folders starting with _)
echo "=== Cleaning .ipynb files ==="
IPYNB_FILES=($(find "$SOURCE_DIR" -name '*.ipynb' -not -path "*/\\_*/*" -not -path "*/.ipynb_checkpoints/*"))
if [[ ${#IPYNB_FILES[@]} -gt 0 ]]; then
    for ipynb_file in "${IPYNB_FILES[@]}"; do
        safe_remove "$ipynb_file" "file"
        ((total_cleaned++))
    done
else
    echo "No .ipynb files found"
fi

echo ""

# 2. Clean .ipynb_checkpoints directories
echo "=== Cleaning .ipynb_checkpoints directories ==="
CHECKPOINT_DIRS=($(find "$SOURCE_DIR" -name '.ipynb_checkpoints' -type d))
if [[ ${#CHECKPOINT_DIRS[@]} -gt 0 ]]; then
    for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
        safe_remove "$checkpoint_dir" "directory"
        ((total_cleaned++))
    done
else
    echo "No .ipynb_checkpoints directories found"
fi

echo ""

# 3. Clean build directory if it exists
echo "=== Cleaning build directory ==="
if [[ -d "$BUILD_DIR" ]]; then
    if [[ "$DRY_RUN" == true ]]; then
        echo "Would delete build directory: $BUILD_DIR"
    else
        echo "Deleting build directory: $BUILD_DIR"
        rm -rf "$BUILD_DIR"
        if [[ $? -eq 0 ]]; then
            echo "  ✓ Deleted successfully"
        else
            echo "  ✗ Failed to delete"
        fi
    fi
    ((total_cleaned++))
else
    echo "No build directory found"
fi

echo ""

# 4. Clean any Python cache files
echo "=== Cleaning Python cache files ==="
PYCACHE_DIRS=($(find "$SOURCE_DIR" -name '__pycache__' -type d))
if [[ ${#PYCACHE_DIRS[@]} -gt 0 ]]; then
    for pycache_dir in "${PYCACHE_DIRS[@]}"; do
        safe_remove "$pycache_dir" "directory"
        ((total_cleaned++))
    done
else
    echo "No __pycache__ directories found"
fi

echo ""

# Summary
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete! Found $total_cleaned items that would be deleted."
    echo "Run without --dry to actually delete these files."
else
    echo "Cleanup complete! Processed $total_cleaned items."
fi