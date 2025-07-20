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


def compile_notebooks(force=False, files=None):
    """Convert .md files to .ipynb notebooks."""
    print("📝 Starting notebook compilation...")
    
    if files:
        # Process specific files provided as arguments
        md_to_process = files
        print(f"📋 Processing {len(md_to_process)} specified files")
    else:
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
        return
    
    # Convert all needed .md files to .ipynb in one batch
    print(f"📝 Converting {len(md_to_process)} markdown files to notebooks...")
    run_command([
        "python", "-m", "jupytext", 
        "--to", "notebook"
    ] + md_to_process)
    print("✅ Notebook compilation completed")


def run_notebooks(force=False, files=None):
    """Execute .ipynb notebooks, deleting those that fail."""
    print("⚡ Starting notebook execution...")
    
    if files:
        # Process specific files provided as arguments
        nb_files = [str(Path(f).with_suffix('.ipynb')) for f in files]
        print(f"📋 Processing {len(nb_files)} specified notebooks")
    else:
        # Find all .ipynb files that have corresponding .md files
        all_md_files = glob.glob("source/**/*.md", recursive=True)
        all_nb_files = [str(Path(md_file).with_suffix('.ipynb')) for md_file in all_md_files]
        
        if force:
            # Process all notebooks when force is enabled
            nb_files = [nb for nb in all_nb_files if Path(nb).exists()]
            print(f"🔄 Force mode: Processing all {len(nb_files)} existing notebooks")
        else:
            # Only process notebooks that exist (skip missing ones)
            nb_files = [nb for nb in all_nb_files if Path(nb).exists()]
            print(f"📋 Found {len(nb_files)} existing notebooks to execute")
    
    if not nb_files:
        print("✅ No notebooks found to execute.")
        return
    
    # Execute notebooks individually and delete if execution fails
    print(f"⚡ Executing {len(nb_files)} notebooks individually...")
    
    failed_files = []
    for nb_file in nb_files:
        print(f"   Executing: {nb_file}")
        result = run_command([
            "python", "-m", "jupyter", "nbconvert",
            "--execute", "--inplace", nb_file
        ], check=False)
        
        if result.returncode != 0:
            print(f"   ❌ Execution failed for {nb_file} - deleting notebook")
            Path(nb_file).unlink(missing_ok=True)
            failed_files.append(nb_file)
        else:
            print(f"   ✅ Successfully executed {nb_file}")
    
    if failed_files:
        print(f"⚠️  {len(failed_files)} notebooks failed execution and were deleted:")
        for failed_file in failed_files:
            print(f"   - {failed_file}")
        print("   These will be recompiled on next compile run.")
    
    successful_count = len(nb_files) - len(failed_files)
    print(f"✅ Notebook execution completed: {successful_count}/{len(nb_files)} successful")


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
  uvx pymoo-docs compile                         # Convert missing .md to .ipynb
  uvx pymoo-docs compile --force                 # Force regenerate all notebooks
  uvx pymoo-docs compile file1.md file2.md      # Convert specific files
  uvx pymoo-docs compile algorithms/nsga2.md    # Convert single file
  uvx pymoo-docs run                             # Execute existing notebooks
  uvx pymoo-docs run --force                     # Force re-execute all notebooks
  uvx pymoo-docs run file1.md file2.md          # Execute specific notebooks
  uvx pymoo-docs build                           # Build HTML documentation
  uvx pymoo-docs serve                           # Serve documentation locally
  uvx pymoo-docs check                           # Fast build for testing
  uvx pymoo-docs all                             # Compile, run, and build
  uvx pymoo-docs all --force                     # Force full rebuild
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
        "files",
        nargs="*",
        help="Specific files to compile (optional)"
    )
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Execute .ipynb notebooks")
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution of all notebooks"
    )
    run_parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to run (optional)"
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
    all_parser = subparsers.add_parser("all", help="Compile, run, and build")
    all_parser.add_argument(
        "--force",
        action="store_true",
        help="Force compilation and execution of all files"
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
    if args.command in ["compile", "run", "build", "check", "all"]:
        install_pymoo()
    
    # Execute the requested command
    if args.command == "clean":
        clean_docs()
    elif args.command == "compile":
        compile_notebooks(force=args.force, files=args.files)
    elif args.command == "run":
        run_notebooks(force=args.force, files=args.files)
    elif args.command == "build":
        build_html()
    elif args.command == "serve":
        serve_docs(args.port)
    elif args.command == "check":
        check_mode()
    elif args.command == "all":
        compile_notebooks(force=args.force)
        run_notebooks(force=args.force)
        build_html()
        print("🎉 Full documentation build completed!")


if __name__ == "__main__":
    main()