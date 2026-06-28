# pymoo — Recommended Improvements (breaking vs non-breaking)

> Companion to [`.claude/docs/ANALYSIS.md`](.claude/docs/ANALYSIS.md), which contains
> the evidence (`file:line`) behind every item here. This document is the **action
> plan**. Each item is framed around what the **end user of the framework** —
> someone who `pip install pymoo`, defines a problem, runs `minimize`, reads the
> result — experiences *today*, what changes, and whether it **breaks existing user
> code**. Finding IDs (C1, H2, M3, …) refer back to ANALYSIS.md.
>
> **Audience:** the maintainer (sequencing + release decisions) and contributors
> (which item to pick up). It is *about* end users; end users don't read it.
>
> **Status:** this plan was reviewed by a multi-advisor council (2026-06-28). The
> sequencing, the breaking/non-breaking labels, and the deprecation-cycle scope were
> corrected as a result; the load-bearing findings it depends on
> (`benchmark/` absent, `mypy.ini` checks disabled, `pyclawd` absent from CI,
> `testing.yml` on `DEACTIVATED`, publish-on-any-tag) were re-verified against the
> tree before encoding the fixes below.

---

## How to read this

### What "breaking" means here (precise)

**Breaking** = existing, *correct* user code can change behavior or stop working after
upgrading. **Non-breaking** = purely additive, or only affects code that was already
crashing / already relying on undefined behavior.

One subtlety the labels below take seriously: **adding a warning is not free.** A new
`UserWarning`/`RuntimeWarning` is visible by default (good — users see NB-2/NB-3/NB-4),
but it **breaks any user who runs with `-W error` or `filterwarnings = error`** (common
in hardened CI). So new warnings are tagged **soft-breaking**: safe under default
filters, breaking under strict ones. A `DeprecationWarning` is the opposite — **hidden
by default**, so it won't reach most users at all (NB-10), which is why deprecations
also need a changelog + upgrade-guide entry, not just a warning.

### The deprecation cycle — applied *only* where it's earned

For changes that alter a **behavioral contract valid old code relied on** (silent inf,
RNG default, result shapes), use:

```
0.6.x (patch/minor)  →  add the new behavior behind a WARNING; old behavior still works
0.7.0 (minor)        →  flip the default / raise the error; warning becomes the rule
```

**Do not** wrap pure bugfixes, dead-code removal, infra, or install corrections in a
cycle — they have no valid old behavior to preserve; ship them immediately. (This was a
mistake in the first draft of this plan: H1, VoidEvaluator, H4 are not deprecations.)

### Recommended execution order (overrides top-to-bottom reading)

The item IDs (NB-/P-/B-) are stable labels, **not** the order to do them in. The
ANALYSIS through-line is that *nothing is gated in CI* — so shipping behavior changes
before a CI net repeats the empty-docs incident. Correct order:

```
Batch 0  (same day)   mechanical quick-wins that need no CI net          → see Part 0
Then     (1–2 wks)    P-1/P-2/P-3 — stand up the CI + docs gate          → Part 2
Then     (0.6.3)      NB behavior/reliability changes, now CI-validated  → Part 1
Then     (0.7.0)      B breaking changes, each pre-warned in 0.6.x       → Part 3
```

This resolves the real tension between "ship user fixes now" and "gate first":
*mechanical* wins ship today; *behavior* changes wait for the net, because they are
exactly what an untested release gets wrong.

### Release-at-a-glance

| Release | Theme | User impact |
|---|---|---|
| **Batch 0** (today) | Mechanical quick-wins, no CI needed | Fixed install extra, fixed docs links, dead code gone. |
| **0.6.4** *(do first)* | CI + docs gate (P-1…P-5) | Invisible to users; the net that makes everything below safe. |
| **0.6.3** *(after the net)* | Honesty + reliability (NB-1…NB-14) | New warnings, populated `Result`, hardened data path. |
| **0.7.0** | Warnings → errors; shape contracts (B-1…B-4) | One-time migration, every break pre-warned in 0.6.x. |

> Effort hints below use **S** (≤½ day), **M** (1–3 days), **L** (>3 days) so the
> maintainer can triage — not story points, just a size.

---

# Part 0 — Same-day quick-wins (no CI net required) **[S each]**

Mechanical, low-risk, independently shippable *today* — decoupled from the big plan so
they don't wait on it:

- **NB-5** add the `[visualization]` extra (or delete it from the docs) — 2-line fix.
- Doc rot: `master`→`main` (`installation.md:111`); `cma` version drift in CLAUDE.md
  (`>=3.2.2` → `>=3.4.0`); landing-page grammar.
- **NB-11** delete `VoidEvaluator` + fix the `Evaluator` mutable default (dead code).
- ~~Add the missing `MOPSO_CD` golden baseline~~ — **false positive**, verified present
  and passing (2026-06-28). No action; see P-5 for the real residual (a coverage gate).

None of these touch a behavioral contract or need the CI net; they only *reduce* the
surface area of the larger releases.

---

# Part 1 — Non-breaking improvements (target: 0.6.3, **after the CI net lands**)

No *correct* existing user code changes behavior; users gain clarity, warnings, and
reliability. New warnings are **soft-breaking** under `-W error` (see definition above).

### NB-1 — Populate `Result.success` and `Result.message` (H1) **[S]**
**Today:** `res.success` is always `None`; users write `if res.success:` and it's
silently falsy.
**Change:** set both at end of run from termination reason + feasibility.
**Why non-breaking:** the field is currently `None` for everyone; filling it can only
help. Also fix the internal consumer at `util/value_functions.py:670`.
**Caveat (council):** don't let `success` become a softer lie than `None`. `None` means
"never set"; `False` means "ran, failed." Code that today treats `None` as "unknown,
skip" would, after this change, see `False` and may treat a *legitimately* failed run
as corrupt — or vice-versa. So set `success=False` **only** for a real failure (no
feasible solution / aborted), `True` for a clean feasible run, and always populate
`message` with the reason. Pair with validate-*on-read* of a `Result` (don't only
guard at write time) so legacy `None`-results from 0.6.x are still distinguishable.
```python
# After (user code that was always-falsy now works):
res = minimize(problem, algorithm, ("n_gen", 200))
if res.success:                       # now a real bool
    deploy(res.X)
else:
    log.warning(res.message)          # e.g. "No feasible solution found."
```

### NB-2 — Warn (don't silently return garbage) on bad evaluations (H2)
**Today:** forgetting `out["F"]` returns `res.F=[inf]`; an infeasible-only problem
returns `res.X=res.F=None` — both with no signal.
**Change (0.6.3, non-breaking):** emit a `RuntimeWarning` ("`_evaluate` did not set
`out['F']`" / "No feasible solution found; returning least-infeasible") and set
`res.message`. Keep returning the same object for now.
**0.7.0 (breaking, see B-1):** unset required `out["F"]` becomes a hard error.
**Why non-breaking now:** a warning + a populated message doesn't change return values.

### NB-3 — Validate `Problem.__init__`, but warn instead of assert (H2)
**Today:** inverted bounds → message-less `AssertionError` deep in sampling;
`len(xl) != n_var` → cryptic broadcast error later; NaN objectives flow silently.
**Change (0.6.3):** check `xl <= xu`, `len(xl)==len(xu)==n_var`, and NaN objectives at
construction; emit a clear `UserWarning` naming the offending indices. (Promote to
`ValueError` in 0.7.0 — B-2.)
**Why non-breaking:** code with valid problems is unaffected; code with invalid bounds
was already crashing later — it now gets a clear message at definition time.

### NB-4 — Warn on the silent RNG entropy fallback (H3) **[M] — soft-breaking**
**Today:** `random_state=None` silently builds an OS-seeded generator; reproducibility
can break with no signal.
**Change:** when no seed/state is provided to a top-level `minimize`/operator call,
warn **once** (`stacklevel` set, `warnings.warn(..., UserWarning)`) — "no seed set;
results will not be reproducible; pass `seed=`". Thread an explicit generator through
internal call paths.
**Classification:** soft-breaking — seeded users see nothing, but unseeded users running
`-W error` now get an exception. That's acceptable (it surfaces a real reproducibility
hole) but it **is** a behavior change; don't market it as invisible. Emit once, not
per-call, to avoid log spam.

### NB-5 — Add the advertised `[visualization]` extra (H4)
**Today:** `pip install pymoo[visualization]` (in the docs) errors — the extra doesn't
exist.
**Change:** add `visualization = [matplotlib, ...]` to `pyproject.toml` optional-deps
(or remove it from the docs). Additive.
```toml
[project.optional-dependencies]
visualization = ["matplotlib>=3.6", "pyrecorder>=...", ...]
```

### NB-6 — Make the compiled→pure-Python fallback diagnosable (H4, Low)
**Today:** ABI mismatch silently falls back to slow pure-Python with a one-time
`print()` to stdout; `is_compiled()` swallows the real exception.
**Change:** route the notice through `warnings.warn` (stderr, filterable), and expose
the underlying import error via a debug helper. Add a `pymoo.show_compile_status()` or
surface it in `minimize(verbose=True)`.
**Why non-breaking:** same fallback behavior, just observable.

### NB-7 — Align build-time numpy with the runtime floor (H4) **[S] — packaging change**
**Today:** wheels compiled against numpy 2.x silently fail to load on numpy-1.x envs →
slow fallback.
**Change (prefer the non-breaking arm):** pin **build** numpy to the ABI-compatible
floor (`oldest-supported-numpy` pattern) so a single wheel loads on both numpy 1.x and
2.x. The API is unchanged → genuinely non-breaking.
**Avoid the breaking arm unless necessary:** *raising the runtime floor* (`numpy>=2.0`)
is a **constraint tightening** — it silently excludes users pinned to numpy 1.x from
upgrading. If you must, treat it as a minor-version bump with a changelog note, not a
patch. The first draft mislabeled this as flatly non-breaking; it depends which arm.

### NB-8 — Harden the remote `.pf` data path (C2) ⭐ highest reliability ROI
**Today:** cache written into `site-packages` (crashes on read-only installs); no
checksum/timeout/atomicity (corrupt partial files cached forever; MITM feeds wrong
Pareto fronts into IGD/HV).
**Change (all non-breaking to the API):**
1. Default cache to a **user dir** (`platformdirs.user_cache_dir("pymoo")` /
   `$XDG_CACHE_HOME`), keep the package dir read-only fallback. Honor a
   `PYMOO_DATA_DIR` override.
2. Download to a temp file + **atomic rename**; add `timeout=`; on any failure delete
   the partial and raise an actionable error.
3. Verify a **committed SHA-256 manifest** for every `.pf`.
4. Ship the tiny `.pf` set as `package_data` *or* add `pymoo.prefetch_data()` for
   offline/airgapped use.
**Why (mostly) non-breaking:** `problem.pareto_front()` keeps the same signature/return;
it stops crashing and stops trusting unverified bytes. **Two honest caveats (council):**
(1) moving the cache **changes the on-disk location** — a one-time re-download, and it
breaks anyone who hard-coded `site-packages/pymoo/data/` (rare, but real); document the
new path + `PYMOO_DATA_DIR`. (2) Atomicity/checksums fix *local* corruption but **do not
remove the GitHub-raw single point of failure or a poisoned-upstream risk** — if the
canonical `.pf` is wrong, prefetch caches it durably. The checksum manifest must be
generated and **committed from a trusted source**, and the offline-bundle/prefetch arm
is what actually removes the SPOF — keep it in scope, not optional.

### NB-9 — Fix default-operator shared singletons (M1)
**Today:** `NSGA2().output is NSGA2().output` → `True`; ask/tell or
`copy_algorithm=False` share mutable operator state.
**Change:** default the operator args to `None` and instantiate inside `__init__`.
**Why non-breaking:** fixes a latent aliasing bug; `minimize`'s deep-copy already hid it
for the common path, so correct user code is unaffected — buggy sharing is removed.

### NB-10 — Deprecate legacy `n_constr` (M2) **[S]**
**Today:** `n_constr=` is silently remapped to `n_ieq_constr` (with `max()` merge
semantics) — equality-constraint users silently misrouted.
**Change:** emit a deprecation pointing to `n_ieq_constr`/`n_eq_constr`. Keep the remap
working through 0.6.x; remove in 0.7.0 (B-3).
**Visibility caveat (council):** a plain `DeprecationWarning` is **hidden by default**,
so most users will never see it and will be surprised by the 0.7.0 removal. Either use
`FutureWarning` (shown by default) for this user-facing rename, **and/or** back the
deprecation with a changelog entry + a 0.7.0 migration-guide section. A warning alone is
not a migration strategy.

### NB-11 — Delete/repair dead always-wrong surfaces (M2)
- Remove `VoidEvaluator` (`core/evaluator.py:138-139`) or fix its read-only-property
  crash — it's referenced nowhere.
- Fix `Evaluator.__init__` mutable default arg → `None`.
**Why non-breaking:** dead code; nothing depends on it.

### NB-12 — Better error messages (Low + H2 shape hints)
- Shape-mismatch errors (`core/problem.py:270`) should name the attribute to fix
  ("…did you set `n_ieq_constr`?").
- Bare `assert np.all(xu >= xl)` (`operators/sampling/rnd.py:25,53`) → add a message.
- `('n_geNOPE', 3)` termination error should echo the bad key + list valid options.
- Narrow bare `except:` (`core/problem.py:193`, `duplicate.py`, `recorder.py`) so
  `KeyboardInterrupt` and real causes aren't swallowed.
**Why non-breaking:** strictly better messages on already-failing paths.

### NB-13 — Documentation fixes (M5, Low)
- Auto-generate the **API Reference** via `autosummary :recursive:` (or a CI check that
  every `pymoo/algorithms/**` class is listed) so new algorithms can't vanish from
  `api/algorithms.rst` (lists 9 of ~33 today).
- Add `.. meta::` description/keywords to all **problem/operator/visualization** pages
  (0/22, 0/7, 0/9 today) — finish what the algorithm pages started.
- Use stdlib `pickle` in the checkpoint tutorial (or document `pip install dill`).
- Switch the parallelization example to a process `Pool` (or document the GIL/pickling
  tradeoff of the current `ThreadPool`).
- Document `save_history=True` requirement next to `res.history`.
- Document the SOO-vs-MOO and Elementwise-vs-vectorized **shape contracts** (M3) — even
  before changing them.
- Fix `master`→`main` (`installation.md:111`), lead with `setup.py build_ext --inplace`
  over `make compile`, fix landing-page grammar, fix the `cma` version drift in CLAUDE.md.

### NB-14 — Packaging hygiene (Low)
- Exclude `*.pyx`/`*.cpp`/`*.pxd` from the **wheel** (keep `.pyx`/`.pxd` in the sdist) —
  removes ~19% bloat.
- `exclude tests/*` → `prune tests` in `MANIFEST.in`.
- Add lower bounds for `alive_progress`/`Deprecated`; raise ancient `scipy`/`matplotlib`
  floors to a tested baseline; consider conservative upper caps on volatile numerical
  deps.
**Why non-breaking:** smaller wheels, same API.

---

# Part 2 — Process / infrastructure (do FIRST; release as 0.6.4 but land before the NB changes)

These don't touch the API at all, but they are the **most important and most
time-consuming** items, and the whole plan depends on them: they stop broken releases
(like the empty-docs deploy) from reaching users, and they are the net that makes the
0.6.3 behavior changes safe to ship. **This is the real critical path** — budget for it
as the multi-week item, not a "polish" afterthought.

### P-1 — Make CI the real gate (C1, H6) ⭐ highest overall ROI **[L]**
- Add a PR job that runs `pyclawd check` + `examples` + `docs` + **`pyclawd golden`**.
- ⚠️ **`pyclawd` is NOT in `tests/requirements.txt`** (verified) — so today the golden
  tests run under bare pytest, return their arrays to *nobody*, and degrade to a
  `PytestReturnNotNoneWarning` **no-op**. The job MUST `pip install pyclawd` (or run the
  golden comparison explicitly) or P-1 is a *false gate* that reports green while
  checking nothing. This is the single most important line in P-1.
- Run the `long` **convergence** tests in CI (they're the only "did it actually
  converge" assertions and are currently skipped by `-m "not long"`).
- Expand the matrix to `{ubuntu, macos, windows} × {3.10–3.14}` for the real suite.
- Re-enable `testing.yml` triggers (currently `branches: [DEACTIVATED]`); replace
  `setup.py install` with `pip install`.

### P-2 — Gate the publish on tests + tag/version consistency (C1)
- Make `publish_pypi` **depend on** the test job.
- Restrict the tag trigger to a semver pattern (not any `refs/tags/*`).
- Add a step asserting `git tag == pymoo.version.__version__` before upload.

### P-3 — Docs build guardrail against empty pages (C3) ⭐ **[M]**
- Add a post-render validator that **fails non-zero** when a code-bearing notebook page
  renders with zero outputs/images.
- **Don't only check for *empty* (council):** the next incident may be *truncated* or
  *garbage* output, not zero bytes. The validator should assert each code cell that
  *should* produce output actually has a **non-empty** one, and ideally that the page's
  output count matches the executed notebook's — catching partial hydration, not just
  total emptiness. (The first-draft "zero outputs" check would re-ship a half-rendered
  page.)
- Stop swallowing hydration errors at `docs/cli.py:399-404` (distinguish "cache miss"
  from "merge error"; treat merge errors as fatal).
- Make `build` refuse to render un-hydrated notebooks (preflight), or call `run` first
  like `all` does.
- Fail the build verdict on Sphinx ERRORs, not just process returncode.

### P-4 — Restore a meaningful typecheck (H5)
Stop disabling `name-defined`/`attr-defined`/`arg-type` in `mypy.ini`; turn on
`check_untyped_defs`; burn down errors module-by-module. Delete the stale
`files = pymoo/core/algorithm.py` line.

### P-5 — Close the regression-net gaps (M6)
- Add a test that **fails** when `is_compiled()` is `False` in the should-be-compiled CI
  lane (catch silent slow fallback).
- Add a minimal `pytest-benchmark` lane with generous tolerances for hot paths
  (non-dominated sorting, hypervolume) so a 10× slowdown can't ship.
- Add an automated **golden-coverage gate** (every parametrized algorithm has a
  committed baseline). *(The `MOPSO_CD`-specific gap was a false positive — already
  present; the gate is what prevents the next real one.)*
- Convert float-equality asserts → `assert_allclose`; seed the correctness-sensitive
  randomized tests.
- Lint `tests/` and `examples/` (reduced ruleset) — examples are user-facing docs.

---

# Part 3 — Breaking changes (target: 0.7.0, each pre-warned in 0.6.x)

Every break here is first shipped as a **warning in 0.6.3** (Part 1), so by 0.7.0 users
have already seen exactly what to change. Bundle them into one minor release with a
migration guide.

### B-1 — Hard error on unset required `out["F"]` (was NB-2)
**Breaking for:** code that (accidentally) never set `out["F"]` and silently got `[inf]`.
That code was already broken; it now fails loudly.
**Migration:** set `out["F"]` in `_evaluate` (the warning in 0.6.3 names the exact problem).
```python
# Before (silently "worked", returned inf):
def _evaluate(self, x, out, *a, **k):
    cost = expensive(x)            # forgot to assign

# After (required):
def _evaluate(self, x, out, *a, **k):
    out["F"] = expensive(x)
```

### B-2 — `Problem.__init__` raises `ValueError` on invalid bounds/dims (was NB-3)
**Breaking for:** code constructing problems with `xl>xu` or mismatched bound lengths —
again, already-crashing code, now caught earlier with a clear message.
**Migration:** fix the bounds; the 0.6.3 warning already pinpointed the indices.

### B-3 — Remove the `n_constr` alias (was NB-10)
**Breaking for:** pre-0.6 code still passing `n_constr=`.
**Migration:** rename to `n_ieq_constr=` (or `n_eq_constr=` for equality). NB-10's
warning flagged call sites — but because `DeprecationWarning` is hidden by default,
**gate this removal on the migration-guide entry existing**, not just on "we warned."

### B-4 — Consistent result shapes (M3) — **opt-in first, default later**
This is the most user-visible shape change, so stage it carefully:
- **0.6.3 (non-breaking):** document the current SOO(1-D)/MOO(2-D) contract; add an
  opt-in `minimize(..., return_2d=True)` (or a `res.opt`/`res.X2d` accessor) that always
  returns 2-D `X`/`F`.
- **0.7.0 (breaking):** make 2-D the default; provide `res.best` (or scalar accessors)
  for the single-solution case.
**Breaking for:** code that relied on SOO `res.X` being 1-D (`res.X[0]` for the first
variable). **Migration:** index `res.X[0]` → `res.best.X`, or iterate rows uniformly.
**Rationale:** the silent rank switch between SOO/MOO and between
`ElementwiseProblem`/`Problem` is a recurring correctness footgun (M3); one documented
contract removes a whole class of bugs.

> Consider also a `maximize=`/sign convenience on `Problem` (M4). Implement as additive
> (non-breaking) in 0.6.x — default `minimize` semantics unchanged — and only document
> it; no 0.7.0 break needed.

---

# Part 4 — End-user "before vs after" summary

What a framework user actually feels, in order of impact:

1. **"My results were silently wrong/garbage" → loud, actionable errors.**
   Unset `out["F"]`, infeasible problems, inverted bounds, NaN objectives, missing seed
   — all now warn (0.6.3) then error (0.7.0) at the point of the mistake instead of
   producing `inf`/`None`/cryptic deep-stack tracebacks.
2. **"`res.success` was useless" → an honest Result.** `success`/`message` populated;
   no-feasible-solution is stated, not a bare `None`.
3. **"pymoo crashed / was 10× slow on my machine" → robust installs.** Remote data no
   longer crashes on read-only installs, can't cache corrupt files, and is integrity-
   checked; numpy ABI mismatch no longer silently degrades; `[visualization]` extra
   exists; compile status is observable.
4. **"The docs page was blank / I couldn't find the algorithm" → trustworthy docs.**
   The empty-page class of bug is structurally prevented in CI; API reference auto-lists
   every algorithm; search lands on every problem/operator page.
5. **"A new release broke something" → caught before it ships.** Full test + examples +
   docs + golden + multi-OS/Python matrix gates every release; the publish can't fire on
   an untested or mislabeled tag.
6. **One clearly-signposted migration at 0.7.0**, with every break already surfaced as a
   warning during 0.6.x — no silent surprises.

---

## The five highest-leverage moves (if you do nothing else)

These are ranked by **leverage, not release bucket** — they intentionally span Batch 0,
0.6.4, and 0.6.3. Do them in the execution order above (quick-wins → CI net → behavior),
but if you only have appetite for five threads, these are them:

1. **P-1 + P-2 + P-3** *(0.6.4 — do first)* — make CI the real gate (**incl. installing
   `pyclawd` so golden isn't a no-op**), gate the publish on tests + tag/version, add the
   docs empty/truncated-page guardrail. *Stops broken releases at the source; everything
   else is unsafe to ship without it.*
2. **NB-8** *(0.6.3)* — harden `remote.py` (user cache, atomic, timeout, **committed**
   checksum, offline prefetch). *Biggest reliability + integrity win.*
3. **NB-1 + NB-2** *(0.6.3)* — honest `Result` (without the `None`→`False` foot-gun) +
   warn-then-error on bad evaluations. *Kills the worst "silently wrong" failure mode.*
4. **NB-3 → B-2** — validate `Problem.__init__`. *Turns the most common beginner
   mistakes into clear messages.*
5. **NB-5** *(Batch 0, today)* — fix the `[visualization]` install lie. *2-line change,
   stops day-one confusion; the cheapest item on the list.* (NB-7's numpy fix is close
   behind but is a packaging-constraint judgment call — see its caveat.)

---

# Appendix — Bigger bets (out of scope for this triage; higher risk, lower reversibility)

The council's Expansionist surfaced structural moves that would subsume several findings
at once. They are **deliberately not** in the breaking/non-breaking plan above: each is
a large, low-reversibility rewrite with a worse completion base-rate than the targeted
fixes, and bundling them would sink the quick wins. Capture them as separate design
proposals, decide independently, and do **not** let them block 0.6.3/0.6.4:

- **`Problem` as a validated spec** (pydantic-v2 / `@dataclass` + field validators, à la
  scikit-learn's `check_estimator`) — would subsume the `__init__` validation (NB-3/B-2),
  the god-object smell, and auto-generate doc schemas. Big surface change to a core class.
- **`Result` as a rigorous container** ("scipy `OptimizeResult` done right") — would make
  H1 and the SOO/MOO shape contract (M3/B-4) first-class fields. Breaking by nature.
- **A storage/cache abstraction** (Optuna-style tiered `prefetch → package → user cache →
  online`, pluggable) — generalizes NB-8 and gives a first-class offline mode.
- **`conda-forge` distribution** — sidesteps both the binary-wheel/numpy-ABI problem
  (NB-7) and the runtime-data problem (ships `.pf` with the package).
- **`towncrier` changelog fragments** enforced to match `__version__` in CI — would make
  P-2's tag/version guard and the deprecation-visibility problem (NB-10) structural
  rather than manual.

**Rule:** none of these is a prerequisite for the plan above. If one is adopted, it
*replaces* its corresponding targeted fix; it does not gate it.

---

## Glossary (shared with ANALYSIS.md)

- **`out["F"]`** — inside a `Problem._evaluate`, the user writes objective values into
  `out["F"]` (and constraints into `out["G"]`/`out["H"]`).
- **SOO / MOO** — single- / multi-objective optimization. Their `Result` shapes differ.
- **`ElementwiseProblem` vs `Problem`** — evaluate one solution at a time (`x` is 1-D)
  vs the whole population vectorized (`X` is 2-D).
- **Pareto front / `.pf`** — the set of optimal trade-off solutions; `.pf` text files are
  the *known* fronts used to score algorithms. Fetched at runtime by `remote.py`.
- **IGD / HV** — Inverted Generational Distance / Hypervolume: quality indicators that
  compare results against the Pareto front (so a wrong `.pf` corrupts them).
- **golden oracle** — a regression test that captures an algorithm's numeric output and
  fails if it drifts from a committed baseline (`@pytest.mark.golden`).
- **`pyclawd`** — the external dev-task CLI driving this repo's test/lint/docs/golden
  commands. (Not a pymoo runtime dependency.)
- **hydration** — merging executed notebook outputs back into the `.ipynb` before Sphinx
  renders; the docs pipeline renders whatever is on disk (`nbsphinx_execute='never'`).
- **the empty-docs incident** — 2026-06-28: 13 of the slowest doc notebooks deployed to
  pymoo.org with empty output cells because render raced ahead of hydration (the seed of
  finding C3 / fix P-3).
- **soft-breaking** — safe under default warning filters; raises under `-W error` /
  `filterwarnings = error`.
