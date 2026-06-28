# pymoo 0.6.2 — End-to-End Release Weakness Analysis

> Scope: a full-stack audit of the framework as shipped in 0.6.2 — core code, the
> public API, packaging/release/install, documentation + docs build pipeline,
> end-user experience, and testing/CI/quality. Every finding is grounded in the
> actual source with `file:line` evidence. Findings independently surfaced by more
> than one reviewer are marked **[cross-validated]** and are high-confidence.
>
> Companion document: [`/IMPROVEMENTS.md`](../../IMPROVEMENTS.md) — the actionable
> recommendations, classified breaking vs non-breaking, sequenced into releases. The
> **glossary** (golden oracle, `pyclawd`, `.pf`/Pareto front, IGD/HV, SOO/MOO, `out["F"]`,
> hydration, the empty-docs incident) lives at the end of that file.
>
> **Audience:** the maintainer and contributors — *not* end users. It describes problems
> users *hit*; it is not user-facing documentation.
>
> **Verification status (2026-06-28):** produced by parallel read-only review agents over
> the 0.6.2 tree, then the load-bearing single-reviewer claims were re-checked directly
> against the repo (prompted by a council review). Confirmed by that pass: `benchmark/`
> is **absent**; `mypy.ini` disables `name-defined`/`attr-defined`/`arg-type` +
> `check_untyped_defs=False`; **`pyclawd` is absent from `tests/requirements.txt`** (so
> the golden oracle no-ops in CI); `testing.yml` is on `branches: [DEACTIVATED]`;
> `build.yml` publishes on any `refs/tags/*`. These are now facts, not claims.

---

## The through-line

The single most important finding ties the whole release together:

**Almost every gate that proves pymoo works runs only when a human types a local
`pyclawd` command.** CI on PRs runs a weak, single-version pytest subset that
*skips* convergence tests, examples, docs, and silently no-ops the golden oracle.
The empty documentation pages discovered and fixed on 2026-06-28 (13 of the slowest
notebooks deployed with zero output cells) were not bad luck — they are the
predictable symptom of this gap. Until the gate moves into CI, every release depends
on a maintainer remembering to run, and not race, the right commands.

Everything below is either downstream of that, or independent correctness/UX debt.

---

## Severity legend

| Tier | Meaning |
|---|---|
| 🔴 Critical | Can ship a broken/incorrect release, corrupt user results, or crash on common installs |
| 🟠 High | Wrong/dishonest API behavior or a real reliability/quality hole users hit |
| 🟡 Medium | Ergonomic sharp edges, silent footguns, coverage gaps |
| ⚪ Low | Polish, doc rot, hygiene |

---

## 🔴 Critical — release integrity & data reliability

### C1. CI does not gate releases **[cross-validated]**
- `.github/workflows/testing.yml` runs `pytest -v --no-header -m "not long"` on a
  single runner (`ubuntu-latest`, Python `3.10`). It never invokes `pyclawd`, so
  format-check, lint, typecheck, descriptions, **examples, docs, and golden** never
  run in CI.
- `push` trigger is `branches: [DEACTIVATED]` — effectively only PR + a weekly cron.
- `.github/workflows/build.yml` trusted-publishes to PyPI on **any** pushed tag
  (`if: startsWith(github.ref, 'refs/tags/')`) with **no test suite** in front of it
  — the only gate is the cibuildwheel `is_compiled()` smoke check (`build.yml:72-73`).
- Tags are lightweight; the package version is `dynamic` from `pymoo/version.py`, so
  the tag name and the published version are independent. Tagging `0.6.3` while
  `version.py` still says `0.6.2` republishes the wrong number.
- **Why it matters:** a single mistaken `git push <tag>` can publish an untested or
  mislabeled wheel to PyPI. Everything that proves correctness is human-dependent.

### C2. The remote `.pf` data path is the biggest reliability + integrity risk
`pymoo/util/remote.py` (server in `pymoo/config.py:25-27`, `raw.githubusercontent.com/anyoptimization/pymoo-data`):
- **Cache is written into the installed package dir** — `remote.py:28`
  `folder = join(dirname(dirname(abspath(__file__))), "data")` →
  `site-packages/pymoo/data/`. On hardened/system-Python/Docker-non-root/Nix
  installs this is read-only, so the first `problem.pareto_front()` raises
  `PermissionError`/`OSError` on a path users assume is offline-safe.
- **No checksum, no timeout, no atomic write** — `remote.py:64-65`
  `urllib.request.urlretrieve(url, f)`. An interrupted download (Ctrl-C, drop, a 404
  HTML body) leaves a truncated file at the final cache path; `load()` only checks
  `os.path.exists(f)` (`remote.py:57`), so the corrupt file is treated as valid
  **forever** until manually deleted. A MITM/compromised response silently feeds
  wrong Pareto fronts into IGD/HV indicator scores — corrupting research results
  with no error.
- **Single point of failure, no offline story** — ~25 problems (`tnk`, `osy`,
  `dtlz5-7`, `MW*`, `ZCAT`, `DASCMOP`, `bbob`, …) call `Remote.get_instance().load()`
  for `pareto_front()`. If GitHub raw is down/rate-limited/renamed, or the box is
  airgapped, all of them break mid-experiment. The `.pf` files are tiny text and are
  shipped neither in the wheel nor the sdist; there is no prefetch command.
- **Doc/code drift:** the release runbook (`SKILL.md:185-186`) describes the download
  as from "the docs host," while the code uses GitHub raw.

### C3. Docs pipeline has no guardrail against empty-output deploys
The pipeline is `.md → jupytext → .ipynb → execute (cached) → hydrate .ipynb → Sphinx
render`. Sphinx is `nbsphinx_execute='never'` (`docs/source/conf.py:158`) — it
renders whatever output state is on disk, executing nothing. Three independent gaps
stack into one silent-deploy bug:
- **No post-render validator** — `docs/cli.py:482 build_html()` / `:444 _summarize_render()`
  derive the verdict only from Sphinx's warning stream and returncode. A page that
  renders with **zero output cells produces no warning and does not change the exit
  code**, so the build reports "✅ clean" and deploys.
- **Hydration errors are swallowed** — `docs/cli.py:399-404`:
  ```python
  for nb in nb_files:
      try:
          cache.merge_match_into_file(nb)
      except Exception:
          pass  # leaves the .ipynb output-less
  ```
- **`build` renders without ensuring hydration** — `docs/cli.py:150-155`: the `build`
  subcommand calls only `build_html()`, unlike `all` (`:166-173`) which does
  compile→run→build and gates on `n_fail`. After a `compile` but before a successful
  `run`, `build` renders never-executed notebooks = empty pages.
- **This is exactly the 2026-06-28 incident** and will recur without a structural fix.

---

## 🟠 High — correctness & honesty of the API

### H1. `res.success` / `res.message` are always `None` **[cross-validated]**
`pymoo/core/result.py:22-23` initialize them; nothing in the run ever assigns them.
Users mimic scipy's `OptimizeResult` and write `if res.success:` → silently always
falsy. pymoo's own `pymoo/util/value_functions.py:670` does
`success = res.success and res.constr_violation <= 0` — always `None`.
*Fix nuance (see IMPROVEMENTS NB-1):* when populating this, don't collapse the
`None` ("never set") and `False` ("ran, failed") states — a naive backfill can make a
failed run indistinguishable from a corrupt one. Set `False` only for genuine failure.

### H2. Silent "success" on a misdefined or infeasible problem **[cross-validated]**
The worst onboarding failure mode — a completed run with garbage and no signal:
- **Unset/typo'd `out["F"]`** → `_format_dict` (`pymoo/core/problem.py:278-282`) fills
  any unset return with `np.full(shape, np.inf)`. A `_evaluate` that sets nothing
  returns `res.F=[inf]` with no error/warning. The lowercase typo `out["f"]` collides
  with the `Individual.f` property (`core/individual.py:399`) and crashes with a
  cryptic `IndexError` that never mentions the typo.
- **Infeasible-only problem** → `pymoo/algorithms/moo/nsga2.py:124`
  `if not has_feasible(self.pop)` returns `res.X = res.F = res.CV = None` silently;
  user hits `TypeError: 'NoneType' object is not subscriptable`, never told "no
  feasible solution found."
- **No `Problem.__init__` validation** — inverted bounds (`xl>xu`) raise a
  **message-less** `AssertionError` deep in `operators/sampling/rnd.py:53`; bound
  length `!= n_var` (`core/problem.py:132-141`) is accepted and later explodes as a
  cryptic NumPy broadcast error far from the cause; NaN objectives propagate into
  selection/survival unless `replace_nan_values_by` is set (`core/problem.py:188-189`).

### H3. RNG reproducibility has a silent entropy fallback
`pymoo/util/__init__.py:31-34`: when `random_state is None`, the
`default_random_state` decorator builds `np.random.default_rng(None)` (fresh,
OS-entropy-seeded) and proceeds with **no warning**. Every operator is wrapped this
way. If any internal path forgets to thread `random_state=` through the `**kwargs`
plumbing, reproducibility breaks silently — defeating the entire 0.6.x RNG refactor.

### H4. Packaging install contract is broken/risky **[cross-validated]**
- **Advertised extra does not exist** — `docs/source/installation.md:68` and
  `.claude/CLAUDE.md` tell users `pip install pymoo[visualization]`, but
  `pyproject.toml` defines only `dev`, `parallelization`, `others`, `full`. The
  command errors (`WARNING: pymoo does not provide the extra 'visualization'`).
- **Build-time numpy unpinned vs runtime floor `numpy>=1.19.3`** —
  `pyproject.toml:38` builds the Cython extensions against latest numpy (2.x); a user
  whose env resolves numpy 1.x fails to import the compiled `.so` (C-ABI mismatch).
  Because `is_compiled()` swallows all exceptions (`pymoo/functions/__init__.py:137`),
  it degrades **silently** to slow pure-Python with only a one-time stdout print.

### H5. The typecheck gate is neutered
`mypy.ini` (which wins over `pyproject.toml` because `pyclawd`'s typecheck cmd passes
no `--config-file`) sets `disable_error_code = misc, arg-type, name-defined,
attr-defined, empty-body, annotation-unchecked` and `check_untyped_defs = False`,
`disallow_untyped_defs = False`. With ~125 `type: ignore` and ~311 `noqa` across
`pymoo/`, the typecheck step is close to a no-op: a refactor that breaks an attribute
access or passes a wrong-typed argument passes the gate green.

### H6. Real convergence tests are `long` → never run in CI
The only genuine "does it reach the known optimum" assertions live in
`tests/algorithms/test_no_error.py` (`assert IGD(pf).do(res.F) < 0.05`;
`assert_almost_equal(fmin, res.F[0], decimal=3)`) and are decorated
`@pytest.mark.long`. CI's `-m "not long"` skips them. Algorithm testing is otherwise
smoke (does-it-raise) + determinism (same seed → same result) + golden (no drift).
A regression that converges to garbage *deterministically* passes determinism and —
because **`pyclawd` is verified absent from `tests/requirements.txt`** — the golden
oracle degrades to a `PytestReturnNotNoneWarning` no-op (`tests/test_golden.py`). This
is the single most important line to fix in CI (IMPROVEMENTS P-1).
So the one numeric tripwire is 100% human-dependent. Examples + docs tests are
**also** marked `long` (`tests/test_examples.py`, `tests/test_docs.py`) → excluded
from CI, which is exactly how the empty-docs shipment happened.

### H7. Correctness matrix is single-OS, single-Python
`testing.yml` pins Ubuntu + Python 3.10. `build.yml` spans
`{windows, ubuntu, ubuntu-arm, macos} × {3.10–3.14}` but its only test is
import + `is_compiled()`. The project claims 3.10–3.14 support (CLAUDE.md), but no
actual test suite runs on 3.11–3.14 or on Windows/macOS — "supported" is asserted,
not verified. cp314 wheels are built although deps (scipy, matplotlib, numba,
moocore) may not yet publish cp314 wheels; nothing installs the full closure to prove
the wheel is usable.

---

## 🟡 Medium — ergonomics, footguns, coverage

### M1. Default operators are shared singletons across instances
Algorithm defaults are evaluated once at import (`pymoo/algorithms/moo/nsga2.py` and
GA/MOEAD/NSGA3/…): `crossover=SBX(...)`, `mutation=PM(...)`,
`output=MultiObjectiveOutput()`. Verified: `NSGA2().output is NSGA2().output` →
`True`; same for `mating.crossover`. `minimize()` masks this via deep-copy, but
`copy_algorithm=False`, manual ask/tell, or sharing an operator across two algorithms
shares mutable parameter-control/display state.

### M2. Several dead / always-wrong API surfaces
- `VoidEvaluator` (`pymoo/core/evaluator.py:138-139`) sets `individual.feas = [False]`,
  but `feas` is a read-only property (`core/individual.py:413-420`) → guaranteed
  `AttributeError`. Dead (referenced nowhere) but crashes if instantiated.
- `Evaluator.__init__` mutable default arg `evaluate_values_of=["F","G","H"]`
  (`evaluator.py:16`) — latent aliasing bug.
- Deprecated `n_constr` kwarg is silently remapped to `n_ieq_constr` with **no
  `DeprecationWarning`** and surprising `max()` merge semantics
  (`core/problem.py:83-87`); equality-constraint users are silently misrouted to
  inequality.

### M3. Result-shape and problem-mode contracts are silent
- **SOO vs MOO** — single-objective `res.X` is 1-D `(n_var,)` / `res.F` is `(1,)`;
  multi-objective `res.X` is `(n_sol, n_var)` / `res.F` is `(n_sol, n_obj)`
  (`core/algorithm.py:287-288`). Code written for one breaks on the other, undocumented.
- **`ElementwiseProblem` vs vectorized `Problem`** — `x` is 1-D vs 2-D with an
  identical `_evaluate` signature (`core/problem.py:34/358`); `x[0]**2` means
  different things and often "works" by accident with wrong results.

### M4. Real-world usage sharp edges
- **Maximization** requires manual `-1` negation with no flag/warning →
  `out["F"]=profit` silently minimizes (docs note it, code doesn't guard).
- **`save_history=False` by default** but docs point at `res.history` for convergence
  plots → empty list; cost/behavior not surfaced at the `minimize` call site.
- **Parallelization example uses a GIL-bound `ThreadPool`**
  (`examples/problems/parallelization.py:1,23`) → no speedup for CPU-bound `_evaluate`;
  the process-Pool alternative needs picklable problems and re-pickles each batch,
  unmentioned.
- **Checkpoint/resume tutorial uses `dill`** (`docs/source/misc/checkpoint.md`), not a
  core dep → `ModuleNotFoundError` on a fresh install; resume also needs non-obvious
  `copy_algorithm=False` + manual `termination` reset.
- **Verbose columns** are cryptic and context-dependent — `n_nds|igd|gd|hv` with a PF,
  `eps|indicator` without one (`pymoo/util/display/multi.py:39-74`), no legend.

### M5. Documentation coverage & discoverability gaps
- **API Reference is hand-maintained and stale** — `docs/source/api/algorithms.rst`
  lists 9 of ~33 algorithms (`.. autoclass::` fixed list: GA, DE, PSO, NSGA2, RNSGA2,
  NSGA3, UNSGA3, RNSGA3, MOEAD). New algorithms (Omni, GDE3, CTAEA, SMS-EMOA, SPEA2,
  RVEA, CMOPSO, MOPSO-CD, AGEMOEA/2, NRBO, …) never appear. `autosummary` is loaded
  (`conf.py:51`) but unused. *(Per-page `autoclass` does render each algorithm's
  hyperparameters, so content coverage is fine; the aggregated index is the gap.)*
- **Meta keywords missing on non-algorithm pages** — problems 0/22, operators 0/7,
  visualization 0/9 (algorithm pages were completed 2026-06-28). These are high-intent
  search landing pages.
- **Operator/problem reference is conceptual-only** — many implemented variants
  (crossovers `erx, ox, pcx, dex, hux, spx, …`, mutations `inversion, gauss, rm`,
  repairs) have no page or API entry; users must read source to discover them.

### M6. Test/quality hygiene & missing safety nets
- **Nothing fails when extensions aren't compiled** — no test references
  `is_compiled()`; `pyclawd doctor` only WARNs. sdist/source installs silently run the
  slow path.
- **No perf-regression tracking** — the `benchmark/` dir referenced in CLAUDE.md is
  **verified absent**; the only perf-adjacent file (`tests/benchmark/run_native_biobj.py`)
  has no assertions. An algorithm can get 10× slower unnoticed. (Note: CLAUDE.md and
  `pyproject.toml` ruff config both reference `benchmark/` — a stale path to fix too.)
- **No automated golden-coverage gate** — there is no CI check that every parametrized
  algorithm has a committed baseline, so a future addition could silently lack one.
  *(Correction: an earlier draft claimed `MOPSO_CD`'s baseline was missing — re-verified
  2026-06-28, it is present in `tests/golden/test_golden.json` and `test_golden_moo[MOPSO_CD]`
  passes. A false positive; the residual finding is the absent automated gap-check.)*
- **Float-equality asserts** contradict the guide — e.g.
  `tests/algorithms/test_algorithms.py:68` `assert res.CV[0] == 1.0`;
  `tests/algorithms/test_deterministic_mixed.py:68`. Brittle across the wider matrix.
- **Unseeded `np.random`** in correctness-sensitive tests (`test_archive.py:105`,
  `test_decomposition.py`, `indicators/test_hv.py`, …) — latent flakiness.
- **Tests/examples exempt from lint/format** (`pyproject.toml` ruff `extend-exclude`),
  yet examples are user-facing docs.

---

## ⚪ Low — polish & doc rot

- **Compile-fallback notice uses `print()` to stdout** (`pymoo/functions/__init__.py:92-98`),
  not `warnings` — can't be filtered, pollutes every script/notebook on a non-compiled
  install.
- **Opaque termination error** — a typo'd `('n_geNOPE', 3)` raises
  `Exception("Termination not found.")` without echoing the bad key or valid options.
- **Wheels ship `.pyx`/`.cpp` sources** (~19% bloat) — `MANIFEST.in:1-4` has no
  wheel-side exclusion; `exclude tests/*` should be `prune tests`.
- **Broad bare `except:`** swallows everything incl. `KeyboardInterrupt`
  (`core/problem.py:193`, `core/duplicate.py:123,126`, `core/recorder.py:62,84`).
- **Cython `language_level=2`** pinned (`pymoo/functions/compiled/info.pyx:2` + siblings).
- **Stale `master` branch reference** — `docs/source/installation.md:111` says
  "master"; default branch is `main`.
- **Install page tells users `make compile`** (`installation.md:122,141`) — assumes a
  Makefile + `make` (often absent on Windows, right after claiming Windows support).
- **Dependency pins:** `alive_progress` and `Deprecated` have **no version floor**;
  `scipy>=1.1` / `matplotlib>=3` floors are ancient; no upper caps anywhere (the cma<3.4
  / NumPy-2 break was reactive); `moocore>=0.1.7` is a young single-source dep now on the
  import path. Doc drift: `.claude/CLAUDE.md` says `cma>=3.2.2`, pyproject pins `>=3.4.0`.
- **Landing-page grammar** — `index.rst` "The command above attempts is made to compile…".
- `Problem` is a 16-parameter god-object (`core/problem.py:34-54`) mixing evaluation,
  gradients, PF/ideal/nadir caching, serialization control, NaN handling, and the
  elementwise runner — long-term maintainability smell.

---

## What is actually healthy (don't over-invest)

- **Module-description migration is essentially done** — `pyclawd ls --missing` →
  324/326 described (the 2 misses are vendor non-Python files).
- **Algorithm content coverage is complete** — every algorithm in
  `pymoo/algorithms/{moo,soo}` has a `.md` page; `algorithms.csv` is current.
- **Onboarding prose is solid** — `getting_started/` has a real Preface + Parts I–V
  (problem → solve → MCDM → convergence → extras) with a clear TOC.
- **The canonical first-run path works cleanly** — `get_problem` + `NSGA2` +
  `minimize` (README 91-107) runs without friction. The pain is concentrated in custom
  problem definition, silent failure modes, and result interpretation — not the happy path.

---

## Cross-validation & confidence notes

- High-confidence (multiple reviewers): **C1, H1, H2, H4, H6**.
- **Re-verified directly against the repo** (2026-06-28, after a council review flagged
  that single-reviewer claims shouldn't be acted on unchecked): `benchmark/` absent (M6);
  `mypy.ini` disables the key checks (H5); `pyclawd` absent from CI test deps (H6);
  `testing.yml` `DEACTIVATED` + `build.yml` publish-on-any-tag (C1). All held.
- **Still worth a glance before acting:** the exact line numbers cited throughout drift
  as the source changes — treat them as "as of 0.6.2 / commit at audit time," and
  re-locate by symbol if they don't match. The MEDIUM/LOW tier was not exhaustively
  re-run finding-by-finding; the CRITICAL/HIGH tier was.
- For a deduplicated, adversarially-checked issue tracker (each finding independently
  refuted-or-confirmed and turned into a GitHub issue), run the verification workflow.
