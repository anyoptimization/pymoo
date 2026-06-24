"""pymoo's pyclawd project configuration.

This file describes the pymoo repository to the `pyclawd` developer toolkit.
pyclawd discovers it by walking up from the current directory to find
``.pyclawd/config.py``, imports it, and reads the module-level ``project``
object below. Every pymoo-specific knob lives here; the pyclawd core stays
project-generic.
"""

from __future__ import annotations

from pyclawd import (
    FAIL,
    OK,
    WARN,
    Check,
    DocsConfig,
    DoctorConfig,
    Project,
    QualityConfig,
    TestConfig,
)


# --------------------------------------------------------------------------- #
# Project-specific doctor checks: pymoo import status + compiled extensions.
# --------------------------------------------------------------------------- #
def pymoo_doctor_checks() -> list[Check]:
    """Return pymoo-specific health checks (import status + compiled extensions)."""
    import importlib
    import os

    checks: list[Check] = []
    try:
        pymoo = importlib.import_module("pymoo")
    except Exception as exc:  # noqa: BLE001 - report any import failure
        return [
            Check(FAIL, "pymoo", f"not importable ({type(exc).__name__})"),
            Check(FAIL, "compiled ext", "n/a — pymoo not importable"),
        ]

    version = getattr(pymoo, "__version__", "?")
    checks.append(Check(OK, "pymoo", f"{version} @ {os.path.dirname(pymoo.__file__)}"))

    try:
        from pymoo.functions import is_compiled

        if is_compiled():
            checks.append(Check(OK, "compiled ext", "Cython extensions active"))
        else:
            checks.append(
                Check(WARN, "compiled ext", "running pure-Python (slow) — `pyclawd compile`")
            )
    except Exception as exc:  # noqa: BLE001
        checks.append(
            Check(WARN, "compiled ext", f"could not determine ({type(exc).__name__})")
        )
    return checks


project = Project(
    name="pymoo",
    conda_env="default",
    root_markers=["pymoo/__init__.py", "setup.py"],
    # Default root for `pyclawd ls` — pymoo uses a flat package layout (not src/).
    src_dir="pymoo",
    # `pyclawd compile` / `pyclawd dist` — args handed to the dev Python.
    compile_cmd=["setup.py", "build_ext", "--inplace"],
    dist_cmd=["setup.py", "sdist"],
    # `pyclawd clean` — root-relative paths to remove.
    clean_targets=["build", "dist", "pymoo.egg-info"],
    # `pyclawd clean --ext` — compiled-artifact dir and the globs removed under it.
    clean_ext_dir="pymoo/functions/compiled",
    clean_ext_globs=["*.c", "*.cpp", "*.so", "*.html"],
    docs=DocsConfig(
        runner=["uvx", "--from", "./docs", "pymoo-docs"],
        source_dir="docs/source",
        cache_dir="docs/.jupyter_cache",
        cache_db="docs/.jupyter_cache/global.db",
        build_html="docs/build/html",
        branch="main",
    ),
    test=TestConfig(
        tests_dir="tests/",
        classname_prefix="tests.",
        integration_files=["tests/test_examples.py", "tests/test_docs.py"],
        markers={
            # Two unit tiers (everything runs under xdist now, so a middle tier
            # buys little): `fast` is the quick smoke (excludes the heavy
            # slow+long tests); `default`/`run`/`all` are the full unit suite.
            # `slow` and `long` therefore both mean "kept out of the fast tier".
            "default": "not examples and not docs",
            "fast": "not examples and not docs and not long and not slow",
            "all": "not examples and not docs",
            "examples": "examples",
            "docs": "docs",
        },
    ),
    quality=QualityConfig(
        lint_cmd=["ruff", "check", "pymoo"],
        lint_fix_cmd=["ruff", "check", "--fix", "pymoo"],
        format_cmd=["ruff", "format", "pymoo"],
        format_check_cmd=["ruff", "format", "--check", "pymoo"],
        typecheck_cmd=["mypy", "pymoo"],
    ),
    doctor=DoctorConfig(
        core_deps=["numpy", "scipy", "matplotlib", "autograd", "cma", "moocore"],
        dev_deps=["pytest", "jupytext", "nbformat", "nbconvert"],
        tool_files=[],  # no wrapper shims — pyclawd is the installed dev command
        binaries=[("pandoc", "conda install -c conda-forge pandoc")],
    ),
    extra_doctor_checks=pymoo_doctor_checks,
)
