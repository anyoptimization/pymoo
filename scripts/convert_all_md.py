#!/usr/bin/env python3
"""
Script to convert all markdown files in docs/source to synced Jupyter notebooks.

This script recursively finds all .md files in the docs/source directory
and converts them to .ipynb files using jupytext with sync enabled.

Usage:
    python convert_all_md.py [--dry-run] [--source-dir <path>]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_markdown_files(source_dir):
    """Find all markdown files in the source directory."""
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        return []
    
    # Find all .md files recursively
    md_files = list(source_path.rglob("*.md"))
    
    print(f"Found {len(md_files)} markdown files in {source_dir}")
    return md_files


def convert_md_to_ipynb(md_file, dry_run=False):
    """Convert a markdown file to a synced Jupyter notebook."""
    if dry_run:
        print(f"[DRY RUN] Would convert: {md_file}")
        return True
    
    try:
        # Use jupytext to create a synced notebook
        cmd = [
            'jupytext',
            '--to', 'ipynb',
            '--sync',
            str(md_file)
        ]
        
        print(f"Converting {md_file}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Check if the ipynb file was created
        ipynb_path = md_file.with_suffix('.ipynb')
        if ipynb_path.exists():
            print(f"✓ Successfully created {ipynb_path}")
            return True
        else:
            print(f"✗ Failed to create {ipynb_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Error converting {md_file}: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ Error: jupytext not found. Please install it with: pip install jupytext")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert all markdown files to synced Jupyter notebooks"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be converted without actually doing it"
    )
    parser.add_argument(
        "--source-dir", 
        default="docs/source", 
        help="Source directory to search for markdown files (default: docs/source)"
    )
    parser.add_argument(
        "--exclude", 
        nargs="*", 
        default=["README.md", ".ipynb_checkpoints"], 
        help="Files to exclude from conversion"
    )
    
    args = parser.parse_args()
    
    # Find all markdown files
    md_files = find_markdown_files(args.source_dir)
    
    if not md_files:
        print("No markdown files found.")
        return
    
    # Filter out excluded files
    excluded_patterns = args.exclude
    filtered_files = []
    
    for md_file in md_files:
        exclude_file = False
        for pattern in excluded_patterns:
            if pattern in str(md_file):
                print(f"Excluding {md_file} (matches pattern: {pattern})")
                exclude_file = True
                break
        
        if not exclude_file:
            filtered_files.append(md_file)
    
    print(f"\nProcessing {len(filtered_files)} files...")
    
    if args.dry_run:
        print("\n--- DRY RUN MODE ---")
    
    # Convert each file
    success_count = 0
    fail_count = 0
    
    for md_file in filtered_files:
        if convert_md_to_ipynb(md_file, args.dry_run):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print(f"\n--- SUMMARY ---")
    print(f"Successfully converted: {success_count}")
    print(f"Failed conversions: {fail_count}")
    print(f"Total processed: {len(filtered_files)}")
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()