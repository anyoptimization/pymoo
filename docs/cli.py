#!/usr/bin/env python3
"""
PyMoo Documentation Builder CLI

This module provides a command-line interface for building pymoo documentation
that can be run with uvx without requiring a dedicated conda environment.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import shutil
import glob


def run_command(cmd, cwd=None, check=True):
    """Run a shell command and stream output to console."""
    print(f"Running: {cmd}")
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    result = subprocess.run(cmd, cwd=cwd, check=False)
    
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    
    return result


def clean_docs():
    """Clean all generated documentation files."""
    print("🧹 Cleaning documentation files...")
    
    # Remove build directory
    build_dir = Path("build")
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"Removed {build_dir}")
    
    # Remove generated notebooks (but keep manually created ones)
    notebook_files = glob.glob("source/**/*.ipynb", recursive=True)
    for nb_file in notebook_files:
        nb_path = Path(nb_file)
        # Check if there's a corresponding .md file
        md_file = nb_path.with_suffix('.md')
        if md_file.exists():
            nb_path.unlink()
            print(f"Removed generated notebook: {nb_file}")
    
    print("✅ Clean completed")


def install_pymoo():
    """Install pymoo in development mode if not available."""
    try:
        import pymoo
        print("✅ PyMoo is available")
        return True
    except ImportError:
        print("📦 Installing PyMoo in development mode...")
        # Try to install pymoo from parent directory
        parent_dir = Path("..").resolve()
        if (parent_dir / "setup.py").exists() or (parent_dir / "pyproject.toml").exists():
            try:
                run_command([
                    "python", "-m", "pip", "install", "-e", str(parent_dir)
                ], check=False)
                import pymoo
                print("✅ PyMoo installed successfully")
                return True
            except Exception as e:
                print(f"⚠️  Warning: Could not install PyMoo: {e}")
                print("Documentation build may fail for API documentation")
                return False
        else:
            print("⚠️  Warning: PyMoo source not found in parent directory")
            return False


def smart_build(force=False):
    """Smart build: find missing notebooks, convert and execute in batches, then build."""
    print("🚀 Starting smart documentation build...")
    
    # Find all .md files recursively
    all_md_files = glob.glob("source/**/*.md", recursive=True)
    
    if force:
        # Process all .md files when force is enabled
        md_to_process = all_md_files
        print(f"🔄 Force mode: Processing all {len(md_to_process)} markdown files")
    else:
        # Only process .md files that don't have corresponding .ipynb files
        md_to_process = []
        for md_file in all_md_files:
            nb_file = Path(md_file).with_suffix('.ipynb')
            if not nb_file.exists():
                md_to_process.append(md_file)
        
        print(f"📋 Found {len(md_to_process)} markdown files without notebooks (out of {len(all_md_files)} total)")
    
    if not md_to_process:
        print("✅ All notebooks already exist. Use --force to regenerate all.")
    else:
        # Step 1: Convert all needed .md files to .ipynb in one batch
        print(f"📝 Converting {len(md_to_process)} markdown files to notebooks...")
        run_command([
            "python", "-m", "jupytext", 
            "--to", "notebook"
        ] + md_to_process)
        print("✅ Notebook conversion completed")
        
        # Step 2: Execute all newly created notebooks in one batch
        nb_files = [str(Path(md_file).with_suffix('.ipynb')) for md_file in md_to_process]
        print(f"⚡ Executing {len(nb_files)} notebooks...")
        run_command([
            "python", "-m", "jupyter", "nbconvert",
            "--execute", "--inplace"
        ] + nb_files)
        print("✅ Notebook execution completed")
    
    # Step 3: Build HTML documentation
    print("🏗️  Building HTML documentation...")
    build_html()
    print("🎉 Smart build completed!")


def build_html():
    """Build HTML documentation using Sphinx."""
    print("🏗️  Building HTML documentation...")
    
    # Ensure build directory exists
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    
    # Run Sphinx build directly to build/html
    run_command([
        "python", "-m", "sphinx",
        "-b", "html",
        "-d", "build/doctrees",
        "source",
        "build/html"
    ])
    
    # Copy markdown files to build directory (matching original Makefile behavior)
    print("📋 Copying markdown files...")
    md_files = glob.glob("source/**/*.md", recursive=True)
    for md_file in md_files:
        src_path = Path(md_file)
        # Remove 'source/' prefix from the path
        rel_path = src_path.relative_to("source")
        dest_path = build_dir / "html" / rel_path
        
        # Create destination directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        shutil.copy2(src_path, dest_path)
        print(f"Copied {md_file} -> {dest_path}")
    
    print("✅ HTML build completed")
    print(f"📖 Documentation available at: {(build_dir / 'html' / 'index.html').resolve()}")


def serve_docs(port=8000):
    """Serve the built documentation locally."""
    html_dir = Path("build/html")
    if not html_dir.exists():
        print("❌ No built documentation found. Run 'build' first.")
        sys.exit(1)
    
    print(f"🌐 Serving documentation at http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        run_command([
            "python", "-m", "http.server", str(port)
        ], cwd=html_dir)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")


def check_mode():
    """Build documentation in fast check mode (excluding most content)."""
    print("⚡ Building documentation in fast check mode...")
    
    # Set environment variable for check mode
    os.environ["PYMOO_DOCS_CHECK_MODE"] = "1"
    
    # Build with minimal content
    build_html()
    
    print("✅ Check mode build completed")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PyMoo Documentation Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uvx pymoo-docs clean                           # Clean all generated files
  uvx pymoo-docs compile                         # Convert all .md to .ipynb
  uvx pymoo-docs compile --skip-existing         # Only convert missing notebooks
  uvx pymoo-docs compile --force                 # Force regenerate all notebooks
  uvx pymoo-docs compile file1.md file2.md      # Convert specific files
  uvx pymoo-docs compile algorithms/nsga2.md    # Convert single file
  uvx pymoo-docs build                           # Build HTML documentation
  uvx pymoo-docs serve                           # Serve documentation locally
  uvx pymoo-docs check                           # Fast build for testing
  uvx pymoo-docs all                             # Clean, compile, and build
  uvx pymoo-docs all --skip-existing             # Full build, skip existing notebooks
        """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean all generated files")
    
    # Compile command
    compile_parser = subparsers.add_parser("compile", help="Convert .md to .ipynb")
    compile_parser.add_argument(
        "--force",
        action="store_true",
        help="Force compilation of all files"
    )
    compile_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have notebooks"
    )
    compile_parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to compile (optional)"
    )
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build HTML documentation")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve documentation locally")
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for serve command (default: 8000)"
    )
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Fast build for testing")
    
    # All command
    all_parser = subparsers.add_parser("all", help="Clean, compile, and build")
    all_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have notebooks"
    )
    all_parser.add_argument(
        "--force",
        action="store_true",
        help="Force compilation of all files"
    )
    
    args = parser.parse_args()
    
    # Change to docs directory if we're not already there
    if Path("source").exists() and Path("source").is_dir():
        print("📁 Working in docs directory")
    else:
        # Try to find docs directory relative to current location
        current_dir = Path.cwd()
        docs_dir = None
        
        # Check if we're in pymoo root and docs exists
        if (current_dir / "docs" / "source").exists():
            docs_dir = current_dir / "docs"
        # Check if we're already in a docs subdirectory
        elif (current_dir / "../docs/source").exists():
            docs_dir = current_dir / "../docs"
        # Check if docs is in parent directories
        else:
            for potential_dir in [Path("docs"), Path("../docs"), Path("../../docs")]:
                if potential_dir.exists() and (potential_dir / "source").exists():
                    docs_dir = potential_dir
                    break
        
        if docs_dir:
            os.chdir(docs_dir)
            print(f"📁 Changed to docs directory: {docs_dir.resolve()}")
        else:
            print("❌ Could not find docs directory with 'source' folder")
            print(f"Current directory: {current_dir}")
            print("Please run this command from the pymoo root directory or docs directory")
            sys.exit(1)
    
    # Install pymoo if needed for build commands
    if args.command in ["compile", "build", "check", "all"]:
        install_pymoo()
    
    # Execute the requested command
    if args.command == "clean":
        clean_docs()
    elif args.command == "compile":
        # Legacy compile command - use smart_build with force if specified
        smart_build(force=args.force)
    elif args.command == "build":
        build_html()
    elif args.command == "serve":
        serve_docs(args.port)
    elif args.command == "check":
        check_mode()
    elif args.command == "all":
        clean_docs()
        smart_build(force=args.force)
        print("🎉 Full documentation build completed!")


if __name__ == "__main__":
    main()