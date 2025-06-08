#!/usr/bin/env python3
"""
Script to convert a markdown file to a synced Jupyter notebook using jupytext.

Usage:
    python md_to_ipynb.py <markdown_file>
"""

import sys
import subprocess
from pathlib import Path


def convert_md_to_ipynb(md_file):
    """Convert a markdown file to a synced Jupyter notebook."""
    md_path = Path(md_file)
    
    # Check if the markdown file exists
    if not md_path.exists():
        print(f"Error: Markdown file {md_file} does not exist")
        return False
    
    # Check if it's actually a markdown file
    if md_path.suffix.lower() != '.md':
        print(f"Warning: {md_file} does not have .md extension")
    
    try:
        # Use jupytext to create a synced notebook
        cmd = [
            'jupytext',
            '--to', 'ipynb',
            '--sync',
            str(md_path)
        ]
        
        print(f"Converting {md_file} to notebook...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Check if the ipynb file was created
        ipynb_path = md_path.with_suffix('.ipynb')
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
    if len(sys.argv) != 2 or sys.argv[1] in ['-h', '--help']:
        print("Usage: python md_to_ipynb.py <markdown_file>")
        print("\nConverts a markdown file to a synced Jupyter notebook using jupytext.")
        print("\nExample:")
        print("  python md_to_ipynb.py source/algorithms/soo/ga.md")
        sys.exit(0 if len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help'] else 1)
    
    md_file = sys.argv[1]
    success = convert_md_to_ipynb(md_file)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()