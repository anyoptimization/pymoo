---
name: pymoo-release
description: End-to-end runbook for cutting a new pymoo release — version bump, changelog, TestPyPI dry-run, git tag, PyPI publish via GitHub Actions trusted publishing, GitHub Release, and deploying the documentation. Use whenever releasing a new pymoo version (e.g. "release pymoo 0.6.3", "cut a pymoo release", "publish pymoo to PyPI", "deploy the pymoo docs").
when_to_use: Releasing a new version of pymoo — bump the version, write the changelog, dry-run on TestPyPI, tag + publish to PyPI, create the GitHub Release, or deploy the documentation. The maintainer runbook so nothing is missed.
---

# pymoo-release

The ordered runbook to ship a pymoo version. A release has **four independent
surfaces** (do together or separately):

1. **PyPI** — push a tag → GitHub Actions builds wheels+sdist → trusted-publish
2. **GitHub Release** — `gh release create`
3. **Docs** — build → deploy to pymoo.org + archive.pymoo.org
4. **Repo** — `VERSION x.y.z` commit + tag + changelog

Dev toolkit is **pyclawd** (run Python via `pyclawd python`, never bare `python`).

> Deploy infra (S3 bucket path, CloudFront distribution IDs, AWS account) is
> **maintainer-private and intentionally NOT in this public repo** — those values
> live in the maintainer's private notes. This skill describes the *process* only.

---

## 0. One-time setup (verify if a publish fails)

**PyPI Trusted Publishing (OIDC, no tokens)** — registered on both indexes so the
workflow publishes without secrets. If a publish fails with
`invalid-publisher: ... no corresponding publisher`, the matching publisher isn't
registered — add it with these exact values:

- **PyPI** → https://pypi.org/manage/project/pymoo/settings/publishing/
- **TestPyPI** → https://test.pypi.org/manage/account/publishing/ (pending publisher)

| Field | Value |
|---|---|
| Owner | `anyoptimization` |
| Repository | `pymoo` |
| Workflow name | `build.yml` |
| Environment | `pypi` (PyPI) / `testpypi` (TestPyPI) |

GitHub **Environments** `pypi` and `testpypi` exist (Settings → Environments) and
gate each publish with a manual approval. The two publish jobs live in
`.github/workflows/build.yml`: `publish_pypi` (runs on a tag push) and
`publish_testpypi` (runs on manual `workflow_dispatch`).

---

## 1. Pre-flight — everything green

```bash
pyclawd check          # format · lint · typecheck · descriptions · test
pyclawd golden         # behavior-regression snapshots (algorithm outputs)
pyclawd test examples  # all examples run
pyclawd docs build     # all notebooks execute + HTML renders
```

---

## 2. TestPyPI dry-run (recommended before every real release)

Validate build → publish → install on the sandbox, with a **dev version** so the
real `x.y.z` stays unused on TestPyPI.

1. Set `pymoo/version.py` → `"x.y.z.dev1"`, commit, push to `main`.
2. GitHub → **Actions → Build → Run workflow** (`workflow_dispatch`).
3. After the build, `publish_testpypi` waits for the **`testpypi` environment
   approval** — approve it.
4. Verify in a clean conda env (see §9):
   ```bash
   pip install --no-cache-dir -i https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ pymoo==x.y.z.dev1
   ```
   `--no-cache-dir` avoids pip's stale index cache; TestPyPI can lag ~30–60 s to
   index a fresh upload. Re-test with `.dev2`, … (`skip-existing: true` is set).

---

## 3. Version bump (single source of truth)

`pymoo/version.py` is the ONLY place — `pyproject.toml` is `dynamic = ["version"]`
reading `pymoo.version.__version__`, and `docs/conf.py` reads it too. Supported
Pythons **3.10–3.14** (`requires-python`, classifiers, and the build matrix match).

```python
# pymoo/version.py
__version__ = "x.y.z"
```

---

## 4. Changelog + news

**Changelog** = `docs/source/versions.md`. Match the house style (check older
entries):

```markdown
#### x.y.z [[Documentation](http://archive.pymoo.org/x.y.z/)]

- New Algorithm: ... (#PR)
- Fixed ...
- Documentation: ...
```

- Heading carries `[[Documentation](http://archive.pymoo.org/x.y.z/)]` (resolves
  after §8 deploys `archive/x.y.z/`).
- **Flat bullet list** (no bold sub-sections); terse; PR refs like `(#776)`; lump
  new algorithms into a `New Algorithm:` line.
- `versions.md` is a jupytext source → `pyclawd docs exec versions` (or a full
  `pyclawd docs build`) regenerates the notebook before render.

**Homepage news** (optional): prepend to `docs/source/news.rst` (full list) and
`docs/source/news_current.rst` (homepage block — keep the latest ~3).

---

## 5. Commit (house convention)

Commit message is literally **`VERSION x.y.z`** (uppercase, no `v`):

```bash
git add pymoo/version.py docs/source/versions.md docs/source/news*.rst
git commit -m "VERSION x.y.z"
git push origin main
```

---

## 6. Tag → PyPI publish

Tags are **lightweight**, named **`x.y.z`** (NO `v` prefix), on the VERSION commit.
Pushing the tag triggers the **Build** workflow; its `publish_pypi` job (`if: tag`)
does the real upload.

```bash
git tag x.y.z
git push origin x.y.z          # starts the prod build
```

Then GitHub → Actions → the tag-triggered Build run. After wheels+sdist build
(~10–15 min), `publish_pypi` waits for the **`pypi` environment approval** —
approve it, and it trusted-publishes to PyPI.

```bash
gh run list --repo anyoptimization/pymoo --workflow=build.yml --limit 3
gh run rerun <run-id> --repo anyoptimization/pymoo --failed   # re-run just publish, no rebuild
curl -s https://pypi.org/pypi/pymoo/x.y.z/json -o /dev/null -w '%{http_code}\n'  # 200 when live
```

---

## 7. GitHub Release

A pushed tag does NOT auto-create a Release. Create it to match convention
(title `VERSION x.y.z`, body = the changelog bullets, mark Latest):

```bash
gh release create x.y.z --repo anyoptimization/pymoo \
    --title "VERSION x.y.z" --latest --notes-file <changelog-bullets.md>
```

---

## 8. Deploy the docs

Docs are served at **pymoo.org** (live) and **archive.pymoo.org/x.y.z/** (per-version
archive) from an S3 bucket behind CloudFront. Concrete bucket path + distribution
IDs are in the maintainer's private notes (not in this repo).

Build at the FINAL version first (`conf.py` reads `pymoo.__version__`, so do this
after §3 so the site shows `x.y.z`, not a dev tag):

```bash
pyclawd docs build    # -> docs/build/html/  (confirm _static/documentation_options.js shows x.y.z)
```

Then (with the private bucket/IDs):
1. **Back up the current live site** into its own archive folder first — it may be
   newer than the existing snapshot: `aws s3 sync <docs-bucket>/html/ <docs-bucket>/archive/<LIVE_VERSION>/`
   (read the live version from `<docs-bucket>/html/_static/documentation_options.js`).
2. **Replace live**: `aws s3 sync docs/build/html/ <docs-bucket>/html/ --delete`
3. **Archive new**: `aws s3 sync docs/build/html/ <docs-bucket>/archive/x.y.z/`
4. **Invalidate** both CloudFront distributions (live + archive): `aws cloudfront create-invalidation --distribution-id <id> --paths "/*"` (~5–15 min to propagate).

Runtime data note: problem Pareto-front `.pf` files are NOT shipped in the wheel —
`pymoo/util/remote.py` downloads them on demand from the docs host's `data/` path.

---

## 9. Verify the published package (clean room)

Install the REAL release in a fresh env, from OUTSIDE the repo (so you import the
wheel, not the in-tree source) with `PYTHONPATH` cleared:

```bash
conda create -y -n pymoo-rel python=3.12
cd /tmp && env -u PYTHONPATH conda run -n pymoo-rel pip install --no-cache-dir pymoo==x.y.z
env -u PYTHONPATH conda run -n pymoo-rel python -c "
import pymoo, os; print(pymoo.__version__, os.path.dirname(pymoo.__file__))
from pymoo.functions import is_compiled; print('compiled:', is_compiled())   # must be True
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.multi import ZDT1
print(minimize(ZDT1(), NSGA2(), ('n_gen',10), seed=1, verbose=False).F.shape)
from pymoo.problems import get_problem
print('remote data:', get_problem('tnk').pareto_front().shape)
"
```

`compiled: True` is the key check — proves the binary wheel installed (not a
pure-python fallback).

---

## Checklist

- [ ] `pyclawd check` / `golden` / examples / docs build green
- [ ] (optional) TestPyPI dry-run with `x.y.z.devN` verified
- [ ] `pymoo/version.py` → `x.y.z`
- [ ] Changelog in `versions.md` (+ news)
- [ ] Commit `VERSION x.y.z` pushed to `main`
- [ ] Tag `x.y.z` pushed → Build run → approve `pypi` → on PyPI (200)
- [ ] GitHub Release `VERSION x.y.z` (Latest)
- [ ] Docs: back up live → deploy `html/` → archive `x.y.z/` → invalidate CDNs
- [ ] Clean-env `pip install pymoo==x.y.z` (`compiled: True`)

---

## Troubleshooting (things that bit us)

- **`invalid-publisher` on publish** → trusted publisher for that index/env isn't
  registered/matching. The failure log prints the OIDC claims (`repository`,
  `workflow_ref`, `environment`) — register a publisher matching them (§0). PyPI
  and TestPyPI are configured separately.
- **`pip` can't find the just-published version** → pip cached the index; add
  `--no-cache-dir`. TestPyPI also lags; confirm via
  `curl https://test.pypi.org/pypi/pymoo/<v>/json`.
- **CMA-ES crashes on NumPy 2.x** (`np.array(..., copy=False)`) → needs
  `cma>=3.4.0` (already pinned).
- **Live docs show a `.devN` version** → docs were built before the version bump;
  bump `version.py` first, then `pyclawd docs build`.
- **Wheels ~19% bigger than needed** → they ship the Cython `.pyx`/`.cpp` sources
  (only the `.so` is used at runtime). Harmless; optional cleanup to exclude them
  from wheels while keeping `.pyx`/`.pxd` in the sdist.
