# pymoo — Agent Guide

pymoo is a Python framework for single- and multi-objective optimization (NSGA-II/III, MOEA/D, …): algorithms, test problems, operators, indicators, visualization. **This file is the single source of truth for AI agents and contributors.**

## Critical rule — how to run Python

**ALWAYS run Python via `pyclawd python`. NEVER call bare `python` / `python -c`.**

```bash
pyclawd python script.py        # run a script
pyclawd python -m pytest ...     # run a module
pyclawd python -c "import pymoo" # quick check
```

`pyclawd python` runs in the conda env `default` and puts the repo root on `PYTHONPATH`. Bare `python` will miss the env and the in-tree source. (`pyclawd` is the external dev toolkit — `pip install pyclawd`; already installed in the `default` env.)

## Dev commands

`pyclawd` is the single entry point for everything — installed on your PATH in the conda env `default` (`pip install pyclawd`).

| Task | Command |
|---|---|
| Health-check the dev env | `pyclawd doctor` |
| Run a python file/module | `pyclawd python <file>` / `pyclawd python -m <mod>` |
| Fast smoke tier (<30s, xdist) | `pyclawd test fast` |
| Full default gate (skips `long`) | `pyclawd test run` / `pyclawd pytest` |
| Tests by category | `pyclawd test default \| examples \| docs` |
| Run one test / file / dir | `pyclawd test -k "<name>"` · `pyclawd test tests/path::test` · `pyclawd test tests/algorithms/` |
| Lint / autofix | `pyclawd lint` (add `--fix`) |
| Format / check-only | `pyclawd format` (add `--check`) |
| Type-check | `pyclawd typecheck` |
| Aggregate quality gate | `pyclawd check` (format-check → lint → typecheck → test) |
| Build / compile Cython | `pyclawd compile` |
| Clean artifacts | `pyclawd clean` (add `--ext` for compiled) |

`pyclawd check` is the canonical "am I done" gate: it runs format-check → lint → typecheck → test, fail-fast. Tests stop-early with `-x`; rerun only failures with `--lf`. Long tests are excluded by default — pass your own `-m` to override. When in doubt, run `pyclawd doctor` first — it catches the common breakages (missing `jupytext`/`nbconvert`, uncompiled Cython, wrong conda env).

## pyclawd architecture (generic toolkit + project config)

- `pyclawd` is an **external, project-generic** package (`pip install pyclawd`) — it knows nothing about pymoo. ALL pymoo specifics (paths, commands, conda env, deps, test markers/tiers, quality commands, doctor checks) live in **`.pyclawd/config.py`** at the repo root: a module-level `project = Project(...)` (with nested `DocsConfig`, `TestConfig`, `DoctorConfig`, `QualityConfig` — frozen dataclasses from `pyclawd`) plus the `extra_doctor_checks=pymoo_doctor_checks` hook (pymoo version + Cython compiled-extension status).
- **The repo root is the directory containing `.pyclawd/`** — `pyclawd` walks up from cwd to find `.pyclawd/config.py` (override via `--config PATH` or `PYCLAWD_CONFIG`).
- **To change pymoo-specific behavior, edit `.pyclawd/config.py`, NOT the toolkit.**
- **Skills** — the `pyclawd-*` skills (`pyclawd-doctor`, `pyclawd-tests`, `pyclawd-quality`, `pyclawd-docs`) install **globally** at `~/.claude/skills/` and shell out to `pyclawd`; they are available in every project, not vendored here.

## Tests, examples, and docs are all tests

- **Unit tests** live in `tests/`, mirroring the package layout. Markers: `long`, `slow`, `examples`, `docs`, `gradient` (`pytest.ini`, `--strict-markers`).
- **`examples/`** files are executed as integration tests (`pyclawd test examples`). Each example must be self-contained and runnable.
  - **Headless note:** a few examples render a *live animation* via pyrecorder/opencv `cv2.imshow` (`dnsga2.py`, `kgb.py`, `tsp.py`, `stream.py`). On a headless box (no `$DISPLAY`) these need a virtual framebuffer. The `pytest-xvfb` dev dep auto-starts one when `$DISPLAY` is unset — so `pyclawd test examples` just works — **but it requires the system `xvfb` binary** (`apt-get install xvfb`, not a pip dep) and the GUI `opencv-python` (pulled in by `pyrecorder`; do **not** install `opencv-python-headless`, which has no `imshow`). If those two examples fail with a Qt/`xcb` "could not connect to display" or `cv2.imshow ... not implemented` error, that's the missing piece — not a code bug.
- **Docs** are `.md` sources converted to Jupyter notebooks via **jupytext**, executed, then built with Sphinx (`pyclawd test docs`). **Never edit generated `.ipynb` directly** — edit the `.md` source.

## Test pipeline (`pyclawd test`)

Mirrors `pyclawd docs`: a logged, instrumented runner with **fast / comprehensive tiers** and a `--lf` fix-loop. The `pyclawd-tests` skill is the full guide.

| Task | Command |
|---|---|
| Fast smoke tier (<30s, xdist `-n auto`) | `pyclawd test fast` |
| Full unit suite (`run`/`all`/bare `test` are identical) | `pyclawd test run` |
| The fix-list (lastfailed cache) | `pyclawd test failures` |
| Debug the next failure (live, `--lf -x`) | `pyclawd test fix` |
| Slowest tests from last run | `pyclawd test timings [--top N]` |
| Select a file/dir/nodeid/keyword | `pyclawd test tests/path::name -x` · `pyclawd test -k nsga2` |

- **Two unit tiers (everything runs under pytest-xdist now).** `fast` = `not slow and not long` (the quick smoke); `run`/`all`/bare `test` are the full unit suite (everything but `examples`/`docs`) — collapsed into one because parallelism made a middle tier pointless, so `slow` and `long` now both just mean "kept out of `fast`". Exclude a heavy module from `fast` with `pytestmark = pytest.mark.slow` at module top (a real marker on the tests — **not** a hardcoded list in `conftest.py`). On a 4-core box: fast ≈15s/602 tests, full ≈48s/721 tests. (`examples` and `docs` are separate integration tiers — `all` does **not** include them.)
- **Logs & timing:** `run`/`fast`/`all` tee full output to a per-run log (+ sibling `.junit.xml`), and the log **ends with the same timing + failure tables and verdict** printed to the console — self-contained, like the docs logs.
- **The fix-loop:** stop-early to fix, full run to verify. `pyclawd test failures` → `pyclawd test fix` (reruns only last-failed within the tier, stops at the first, full traceback) → fix the cause → `pyclawd test fix` again → finally `pyclawd test run`. The lastfailed cache (`.pytest_cache`) drives `--lf`; the `examples`/`docs` suites are deselected by the unit tiers, so `failures` lists their stale entries separately. Common fixes: **float-equality assert → `np.testing.assert_allclose(rtol/atol)`**; **unseeded stochastic test → fix the `seed`**; broken-env mass failures → `pyclawd doctor` + `pyclawd compile`.

## Documentation builds (`pyclawd docs`)

Building docs executes ~126 notebooks (real optimization) then renders HTML. Execution is cached, so you rarely pay full cost.

| Task | Command |
|---|---|
| Build everything (cached) | `pyclawd docs build` |
| Only pages changed vs `main` | `pyclawd docs build --changed` |
| Force full re-execution | `pyclawd docs build --all` |
| What would re-run | `pyclawd docs status` |
| Slowest notebooks (bottlenecks) | `pyclawd docs timings` |
| What failed + why | `pyclawd docs failures [--full]` |
| Debug ONE notebook (direct stacktrace) | `pyclawd docs exec <page>` |
| Execute notebooks only (no HTML) | `pyclawd docs run [pages…]` |
| Render HTML only (no execute) | `pyclawd docs render` |
| Serve built HTML | `pyclawd docs serve` |

**`run` (execute) and `render` (Sphinx HTML) are separate** because execution is cached: fix a render-only problem and re-`render` in seconds without re-executing. `build` = run + render. `build`/`render` **preflight `pandoc`** (the system binary nbsphinx needs) and fail in seconds if it's missing — `pyclawd doctor` checks it too.

- **Caching:** jupyter-cache keyed on **code cells** — prose-only edits never re-execute; only changed code re-runs (in parallel).
- **Git stays clean:** nbsphinx is `execute='never'`; executed `.ipynb` and the cache (`docs/.jupyter_cache/`) are gitignored — **only `.md` sources are committed**. Jupyter Lab opens the hydrated (gitignored) `.ipynb`.
- **Isolated deps:** heavy docs deps live in the `./docs` env (via `uvx`), not pymoo.
- **Fixing failures (the loop):** jupyter-cache caches **only successes**, so failed notebooks stay uncached and the next `pyclawd docs build` re-runs **only them**. To debug a failure, **`pyclawd docs exec <page>`** runs that one notebook directly (no cache, no pool, no log) and prints the **full stacktrace** — fix the `.md`, `exec` again, move to the next. `pyclawd docs failures` lists the whole fix-list. Common fixes: missing dep → add to `docs/pyproject.toml`; per-cell timeout → `PYMOO_DOCS_TIMEOUT`; real bug → edit the `.md`. Parallelism defaults to `cores − 2` (override via `PYMOO_DOCS_JOBS`).

## Conventions

- **google-style docstrings** (ruff `convention = "google"`), with **types only in annotations — never duplicated in the docstring**. The codebase is migrating numpy → google module-by-module; the docs render both via sphinx `napoleon` during the transition (set `napoleon_numpy_docstring = False` once done). Every file also opens with a one-line module description (`pyclawd ls --missing`).
- Match existing patterns before inventing new ones. Core abstractions: `Problem`, `Algorithm`, `Operator`, `Population`, `Individual`, `Result`. Main entry point: `from pymoo.optimize import minimize`.
- Performance-critical code has **Cython** counterparts in `pymoo/functions/`; check `is_compiled()`.
- Adjust numerical tolerances (`atol`/`rtol`) rather than chasing exact float equality in tests.

## Project structure

```
pymoo/                    # main package
├── algorithms/           # optimization algorithms (moo/, soo/)
├── core/                 # core framework classes & interfaces
├── problems/             # test problems (multi/, single/, many/, dynamic/)
├── operators/            # genetic operators (crossover, mutation, selection, sampling)
├── indicators/           # performance metrics
├── visualization/        # plotting
├── gradient/             # gradient utilities
├── constraints/          # constraint handling
├── termination/          # termination criteria
├── util/                 # helpers
└── functions/            # compiled performance-critical functions (Cython)

examples/                 # usage examples — run as integration tests
tests/                    # test suite (mirrors the package layout)
docs/                     # documentation source (.md → notebooks)
benchmark/                # performance benchmarking scripts

.pyclawd/config.py        # pymoo's pyclawd project config (Project object + hooks)
.claude/CLAUDE.md         # this file — the agent guide
```

**Core architecture patterns:**
- **Problems** — base `pymoo.core.problem.Problem`; `ElementwiseProblem` for single-point eval; `MetaProblem` for composition. Define `n_var`, `n_obj`, `n_constr`, `xl`, `xu`.
- **Algorithms** — base `pymoo.core.algorithm.Algorithm`; ask-and-tell pattern; modular interchangeable operators; population-based and single-point.
- **Operators** — inherit `pymoo.core.operator.Operator`; composable crossover/mutation/selection/sampling.
- **Data structures** — `Population` (collection), `Individual` (variables/objectives/constraints), `Result` (run container).

**Import conventions:** main API `from pymoo.optimize import minimize`; algorithms `from pymoo.algorithms.moo.nsga2 import NSGA2`; problems `from pymoo.problems import get_problem`.

## Technology stack

- **Python** 3.9+ (3.9–3.13). **Build:** setuptools + Cython. Distributed on PyPI.
- **Core deps:** numpy (≥1.19.3), scipy (≥1.1), matplotlib (≥3), autograd (≥1.4), cma (≥3.2.2), moocore (≥0.1.7), Cython (compiled extensions).
- **Optional:** parallelization (joblib, dask, ray); dev (pytest, jupyter, pandas, numba); optimization (optuna).
- **User install:** `pip install -U pymoo` · `pip install pymoo[full]` (all features) · `pip install pymoo[visualization]`. From source: `git clone … && cd pymoo && pip install .`.
- **Verify compiled extensions:** `pyclawd python -c "from pymoo.functions import is_compiled; print('compiled:', is_compiled())"`.

## Skills

- The `pyclawd-*` skills (`pyclawd-doctor`, `pyclawd-tests`, `pyclawd-quality`, `pyclawd-docs`) are installed **globally** at `~/.claude/skills/` and shell out to `pyclawd` — available in every project, not vendored here.
