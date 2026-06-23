#!/usr/bin/env python3
"""
PyMoo Documentation Builder CLI

This module provides a command-line interface for building pymoo documentation
that can be run with uvx without requiring a dedicated conda environment.
"""

import argparse
import glob
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("pymoo-docs")


def _setup_logging():
    """Structured logs: `HH:MM:SS  LEVEL  message` (with [notebook] context)."""
    if log.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s", "%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False

# Per-cell execution timeout (s). 30s (nbclient default) is too low for the
# optimization notebooks; raise it so long-but-legitimate cells don't time out.
EXEC_TIMEOUT = os.environ.get("PYMOO_DOCS_TIMEOUT", "300")

# Number of notebooks executed in parallel. jupyter-cache's pool uses
# os.cpu_count() with no CLI knob; we cap it (patched in a shim) to leave the
# machine responsive — default = cores - 2 (min 1), override via PYMOO_DOCS_JOBS.
EXEC_JOBS = os.environ.get("PYMOO_DOCS_JOBS") or str(max(1, (os.cpu_count() or 3) - 2))


def read_timings(cache_dir="."):
    """Map notebook abspath -> last execution time (seconds) from the
    jupyter-cache DB. Empty dict if the cache doesn't exist yet."""
    db = Path(cache_dir) / ".jupyter_cache" / "global.db"
    if not db.exists():
        return {}
    timings = {}
    con = sqlite3.connect(str(db))
    try:
        for uri, data in con.execute("SELECT uri, data FROM nbcache"):
            try:
                secs = json.loads(data or "{}").get("execution_seconds")
            except (ValueError, TypeError):
                secs = None
            if secs is not None:
                timings[uri] = secs
    finally:
        con.close()
    return timings


def print_timings(nb_files):
    """Print a per-notebook execution-time table (slowest first) for nb_files.
    Goes to stdout so it is captured in the run log."""
    timings = read_timings(".")
    rows = sorted(
        ((timings[str(Path(nb).resolve())], nb) for nb in nb_files
         if str(Path(nb).resolve()) in timings),
        reverse=True,
    )
    if not rows:
        return
    total = sum(s for s, _ in rows)
    print(f"\n⏱  Execution time (cached, slowest first) — "
          f"total {total:.1f}s across {len(rows)} notebooks:")
    for secs, nb in rows:
        print(f"   {secs:7.1f}s  {nb}")


def read_failures(cache_dir="."):
    """Map notebook abspath -> traceback (ANSI-stripped) for notebooks whose last
    execution failed. jupyter-cache stores these in nbproject.traceback."""
    db = Path(cache_dir) / ".jupyter_cache" / "global.db"
    if not db.exists():
        return {}
    failures = {}
    con = sqlite3.connect(str(db))
    try:
        q = "SELECT uri, traceback FROM nbproject WHERE traceback IS NOT NULL AND traceback != ''"
        for uri, tb in con.execute(q):
            failures[uri] = re.sub(r"\x1b\[[0-9;]*m", "", tb)
    finally:
        con.close()
    return failures


def _error_line(tb):
    """The most informative line of a traceback (the exception)."""
    errs = [l.strip() for l in tb.splitlines() if re.search(r"(Error|Exception|Timeout\w*):", l)]
    if errs:
        return errs[-1]
    lines = [l for l in tb.splitlines() if l.strip()]
    return lines[-1].strip() if lines else "(unknown)"


def print_failures(nb_files=None):
    """Print a failure summary (notebook + exception) — captured in the run log."""
    failures = read_failures(".")
    if nb_files is not None:
        keep = {str(Path(nb).resolve()) for nb in nb_files}
        failures = {u: t for u, t in failures.items() if u in keep}
    if not failures:
        return
    print(f"\n❌ {len(failures)} notebook(s) failed execution:")
    for uri in sorted(failures):
        name = uri.split("/docs/source/")[-1]
        print(f"   {name}: {_error_line(failures[uri])[:140]}")


def show_failures(full=False):
    """Print failed notebooks and their errors (full tracebacks with --full)."""
    failures = read_failures(".")
    if not failures:
        print("✅ No failed notebooks recorded in the cache.")
        return
    print(f"❌ {len(failures)} failed notebook(s):")
    for uri in sorted(failures):
        name = uri.split("/docs/source/")[-1]
        if full:
            print(f"\n=== {name} ===\n{failures[uri].strip()}")
        else:
            print(f"   {name}: {_error_line(failures[uri])[:140]}")


def show_timings():
    """Print all cached notebook execution times (slowest first)."""
    timings = read_timings(".")
    if not timings:
        print("No cache timings yet — run a build first.")
        return
    rows = sorted(((s, u) for u, s in timings.items()), reverse=True)
    total = sum(s for s, _ in rows)
    print(f"⏱  Notebook execution time (cached) — total {total:.1f}s across {len(rows)} notebooks:")
    src = Path("source").resolve()
    for secs, uri in rows:
        try:
            uri = str(Path(uri).relative_to(src))
        except ValueError:
            pass
        print(f"   {secs:7.1f}s  {uri}")


def run_command(cmd, cwd=None, check=True):
    """Run a shell command and stream output to console."""
    if isinstance(cmd, str):
        cmd = cmd.split()
    print(f"$ {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=cwd, check=False)
    
    if check and result.returncode != 0:
        sys.exit(result.returncode)

    return result


def jcache(args, check=True):
    """Run a `jcache` command, auto-confirming the one-time cache creation prompt.

    The cache lives in `.jupyter_cache/` in the docs directory (cwd). It is
    gitignored — executed outputs live ONLY here and in the gitignored .ipynb,
    never in git.
    """
    cmd = ["jcache"] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, input="y\n", text=True, check=False)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result


def jcache_execute(force=False):
    """Execute outdated notebooks in parallel, capped at EXEC_JOBS workers.

    jupyter-cache's parallel pool sizes itself from os.cpu_count() with no CLI
    knob, so we run jcache in a shim that patches cpu_count — keeping load sane
    on small machines.
    """
    shim = (
        f"import os; os.cpu_count = lambda: {int(EXEC_JOBS)}; "
        "from jupyter_cache.cli.commands.cmd_main import jcache; jcache()"
    )
    cmd = [sys.executable, "-c", shim, "project", "execute",
           "--executor", "local-parallel", "--timeout", EXEC_TIMEOUT]
    if force:
        cmd.append("--force")
    print(f"⚡ Executing {'ALL' if force else 'outdated'} notebooks "
          f"({EXEC_JOBS} parallel, {EXEC_TIMEOUT}s/cell timeout)...")
    result = subprocess.run(cmd, input="y\n", text=True, check=False)
    if result.returncode != 0:
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
            # Convert sources that are new or whose .md changed since the .ipynb
            # was written. Unchanged pages keep their (hydrated) .ipynb untouched.
            md_to_process = []
            for md_file in all_md_files:
                nb_file = Path(md_file).with_suffix('.ipynb')
                if not nb_file.exists() or Path(md_file).stat().st_mtime > nb_file.stat().st_mtime:
                    md_to_process.append(md_file)

            print(f"📋 {len(md_to_process)} changed/new of {len(all_md_files)} markdown files to (re)compile")
    
    if not md_to_process:
        print("✅ All notebooks already exist. Use --force to regenerate all.")
        return
    
    # Convert all needed .md files to .ipynb in one batch
    print(f"📝 Converting {len(md_to_process)} markdown files to notebooks...")
    run_command([
        "python", "-m", "jupytext",
        "--to", "notebook"
    ] + md_to_process)

    # Normalize the kernel to one that exists in the build env. Some sources
    # declare a machine-specific kernel ('default') that isn't registered here,
    # so nbclient fails with NoSuchKernel before running a single cell.
    import nbformat
    fixed = 0
    for md in md_to_process:
        nb_path = Path(md).with_suffix(".ipynb")
        if not nb_path.exists():
            continue
        nb = nbformat.read(nb_path, 4)
        if nb.metadata.get("kernelspec", {}).get("name") != "python3":
            nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
            nbformat.write(nb, nb_path)
            fixed += 1
    if fixed:
        print(f"🔧 Normalized kernel → python3 in {fixed} notebook(s)")
    print("✅ Notebook compilation completed")


def _execute_one(nb_path, timeout):
    """Execute ONE notebook (kernel forced to python3) and write outputs back.
    Returns (ok, seconds, error_or_None). Runs in its own worker process."""
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(nb_path, 4)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    t0 = time.monotonic()
    try:
        NotebookClient(
            nb, timeout=int(timeout), kernel_name="python3",
            resources={"metadata": {"path": str(Path(nb_path).parent)}},
        ).execute()
        nbformat.write(nb, nb_path)
        return True, time.monotonic() - t0, None
    except CellExecutionError as exc:
        nbformat.write(nb, nb_path)  # keep the error output for backtrack
        lines = [l for l in str(exc).strip().splitlines() if l.strip()]
        return False, time.monotonic() - t0, (lines[-1] if lines else "CellExecutionError")[:140]
    except Exception as exc:  # kernel death, timeout, …
        return False, time.monotonic() - t0, f"{type(exc).__name__}: {exc}"[:140]


def run_notebooks(force=False, files=None):
    """Execute notebooks with structured, per-notebook logging.

    We drive execution ourselves (jupyter-cache stays the cache STORE) so each
    notebook gets a `[name] ok/failed` line, the kernel is normalized to python3
    before running (no NoSuchKernel), and the real error is captured for backtrack
    (`pymoo-docs exec <name>`). Only stale notebooks run; successes are cached and
    every notebook is hydrated for nbsphinx + Jupyter Lab. Outputs never reach git.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from jupyter_cache import get_cache

    if files:
        nb_files = [str(Path(f).with_suffix(".ipynb")) for f in files]
    else:
        all_md = glob.glob("source/**/*.md", recursive=True)
        nb_files = [str(Path(md).with_suffix(".ipynb")) for md in all_md]
    # Use absolute paths everywhere so cache URIs stay consistent (the cache
    # stores whatever path it's given; the failures view compares them).
    nb_files = [str(Path(nb).resolve()) for nb in nb_files if Path(nb).exists()]
    if not nb_files:
        log.info("no notebooks to execute (run 'compile' first)")
        return

    cache = get_cache(".jupyter_cache")
    jobs = int(EXEC_JOBS)
    src_root = str(Path("source").resolve()) + os.sep

    # Partition: which notebooks actually need executing? (cache miss = stale)
    stale, n_cached = [], 0
    for nb in nb_files:
        if force:
            stale.append(nb)
            continue
        try:
            cache.match_cache_file(nb)
            n_cached += 1
        except KeyError:
            stale.append(nb)

    log.info("execute · %d stale · %d cached · %d workers · %ss/cell timeout",
             len(stale), n_cached, jobs, EXEC_TIMEOUT)

    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_execute_one, nb, EXEC_TIMEOUT): nb for nb in stale}
        for fut in as_completed(futures):
            nb = futures[fut]
            name = nb.replace(src_root, "")
            ok, secs, err = fut.result()
            if ok:
                cache.cache_notebook_file(nb, data={"execution_seconds": secs}, overwrite=True)
                log.info("[%s] ok · %.1fs", name, secs)
                n_ok += 1
            else:
                log.error("[%s] failed · %s", name, err)
                n_fail += 1

    # Hydrate every notebook from the cache so nbsphinx + Lab show outputs.
    for nb in nb_files:
        try:
            cache.merge_match_into_file(nb)
        except Exception:
            pass  # not cached (a failure) — leave it output-less

    (log.warning if n_fail else log.info)(
        "done · %d ok · %d failed · %d reused", n_ok, n_fail, n_cached)
    return n_fail


def exec_single(page):
    """Execute ONE notebook directly (no cache, no pool) and show its error
    immediately. The debug loop: run one → read the traceback → fix the .md →
    run again. On success, add it to the cache so a later build reuses it."""
    name = page.removesuffix(".ipynb").removesuffix(".md")
    md = f"{name}.md" if name.startswith("source/") else f"source/{name}.md"
    if not Path(md).exists():
        print(f"❌ source not found: {md}")
        sys.exit(2)
    nb = Path(md).with_suffix(".ipynb")

    # Compile + normalize the kernel so execution doesn't trip on NoSuchKernel.
    run_command(["python", "-m", "jupytext", "--to", "notebook", md])
    import nbformat
    n = nbformat.read(nb, 4)
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nbformat.write(n, nb)

    # Execute directly — a cell error prints its FULL traceback to the console.
    print(f"\n⚡ Executing {nb} (timeout {EXEC_TIMEOUT}s/cell)...\n")
    result = run_command([
        "python", "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
        "--inplace", f"--ExecutePreprocessor.timeout={EXEC_TIMEOUT}", str(nb),
    ], check=False)

    if result.returncode == 0:
        print(f"\n✅ {nb} executed cleanly")
        jcache(["cache", "add", str(nb)], check=False)  # feed the build cache
    else:
        print(f"\n❌ {nb} failed — full traceback above. Fix {md} and re-run.")
    sys.exit(result.returncode)


def _summarize_render(warnfile: Path, renderlog: Path, returncode: int) -> int:
    """Turn Sphinx's raw warning stream into a structured render verdict.

    Mirrors the execute step: the full per-page Sphinx output is teed to
    `renderlog` (the long log, like the run log), warnings are captured to
    `warnfile` (via `sphinx -w`), and here we print a deduped, severity-grouped
    summary so a glance at the tail tells you whether the render is clean.
    Returns the number of unique warnings/errors.
    """
    lines = []
    if warnfile.exists():
        lines = [ln.strip() for ln in warnfile.read_text(errors="replace").splitlines() if ln.strip()]

    # Dedupe identical messages, keep first-seen order, and split by severity.
    seen, errors, warnings = set(), [], []
    for ln in lines:
        if ln in seen:
            continue
        seen.add(ln)
        (errors if "ERROR" in ln or "CRITICAL" in ln else warnings).append(ln)

    n = len(errors) + len(warnings)
    if returncode != 0:
        log.error("render · sphinx exited %d — HTML may be incomplete", returncode)
    if n == 0:
        log.info("render · clean · 0 warnings · log %s", renderlog)
        return 0

    emit = log.error if errors else log.warning
    emit("render · %d unique issue(s) · %d error(s) · %d warning(s) · warnlog %s · log %s",
         n, len(errors), len(warnings), warnfile, renderlog)
    for ln in (errors + warnings)[:30]:
        emit("  %s", ln)
    if n > 30:
        emit("  … %d more — see %s", n - 30, warnfile)
    return n


def build_html():
    """Build HTML documentation using Sphinx."""
    print("🏗️  Building HTML documentation...")

    # Ensure build directory exists
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)

    # The render step gets a real log, like the run step:
    #   - renderlog: the FULL per-page Sphinx output, teed to a standalone file
    #     (and still streamed → captured in the pyclawd run log).
    #   - warnfile:  just the warnings/errors (via `sphinx -w`) for the summary.
    # --keep-going lets Sphinx finish and report every issue, not bail on first.
    warnfile = build_dir / "sphinx-warnings.log"
    renderlog = build_dir / "sphinx-render.log"
    cmd = [
        "python", "-m", "sphinx",
        "-b", "html",
        "-d", "build/doctrees",
        "--keep-going",
        "-w", str(warnfile),
        "source",
        "build/html",
    ]
    print(f"$ {' '.join(cmd)}")
    log.info("render · sphinx starting · full log %s", renderlog)
    with open(renderlog, "w") as rf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)  # stream → captured by the pyclawd run log
            rf.write(line)          # standalone long render log
        proc.wait()

    n_warn = _summarize_render(warnfile, renderlog, proc.returncode)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

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
    
    verdict = "✅ HTML build completed" if n_warn == 0 else f"⚠️  HTML build completed with {n_warn} render warning(s) — see {warnfile}"
    print(verdict)
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
  uvx pymoo-docs build --fast                    # Fast build with few sample notebooks
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
    run_parser = subparsers.add_parser("run", help="Execute notebooks (jupyter-cache, parallel, skip-unchanged)")
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
    run_parser.add_argument(
        "--continue", dest="cont", action="store_true",
        help="Don't exit non-zero when notebooks fail",
    )

    # Build command
    build_parser = subparsers.add_parser("build", help="Build HTML documentation")
    build_parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast build: only compile/run a few sample notebooks"
    )
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve documentation locally")
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for serve command (default: 8000)"
    )
    
    # Timings command
    timings_parser = subparsers.add_parser("timings", help="Show cached per-notebook execution times (slowest first)")

    # Failures command
    failures_parser = subparsers.add_parser("failures", help="Show notebooks whose last execution failed, with the error")
    failures_parser.add_argument("--full", action="store_true", help="Show full tracebacks")

    # Exec command — run ONE notebook directly and show its error (debug loop)
    exec_parser = subparsers.add_parser("exec", help="Execute ONE notebook directly and show its error")
    exec_parser.add_argument("page", help="Notebook page, e.g. visualization/pcp")

    # Check command
    check_parser = subparsers.add_parser("check", help="Fast build for testing")
    
    # All command
    all_parser = subparsers.add_parser("all", help="Compile, run, and build")
    all_parser.add_argument(
        "--force",
        action="store_true",
        help="Force compilation and execution of all files"
    )
    all_parser.add_argument(
        "--continue", dest="cont", action="store_true",
        help="Render HTML even if some notebooks failed (default: stop early)",
    )

    args = parser.parse_args()
    _setup_logging()
    
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
    if args.command in ["compile", "run", "build", "check", "all", "exec"]:
        install_pymoo()
    
    # Execute the requested command
    if args.command == "clean":
        clean_docs()
    elif args.command == "compile":
        compile_notebooks(force=args.force, files=args.files)
    elif args.command == "run":
        n_fail = run_notebooks(force=args.force, files=args.files)
        if n_fail and not args.cont:
            sys.exit(1)
    elif args.command == "build":
        if args.fast:
            # Fast build: set environment variable to exclude most files
            os.environ["PYMOO_DOCS_FAST_MODE"] = "1"
            print("⚡ Fast build mode: excluding most directories")
        build_html()
    elif args.command == "serve":
        serve_docs(args.port)
    elif args.command == "timings":
        show_timings()
    elif args.command == "failures":
        show_failures(full=args.full)
    elif args.command == "exec":
        exec_single(args.page)
    elif args.command == "check":
        check_mode()
    elif args.command == "all":
        compile_notebooks(force=args.force)
        n_fail = run_notebooks(force=args.force)
        if n_fail and not args.cont:
            log.error("%d notebook(s) failed — NOT building HTML. "
                      "Fix them (`pymoo-docs failures` / `exec <page>`) or pass --continue.", n_fail)
            sys.exit(1)
        build_html()
        log.info("documentation build complete")


if __name__ == "__main__":
    main()