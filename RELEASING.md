# Releasing the Lattice Grid Python packages

This repo ships three separately-installable packages, released **together**, on
one version number:

| Package | Path | Purpose |
| --- | --- | --- |
| `lattice-grid-pandas` | `packages/lattice-grid-pandas` | Shared, framework-agnostic pandas <-> grid serialization + grid/licence plumbing. |
| `lattice-grid-jupyter` | `.` (repo root) | anywidget notebook widget. |
| `lattice-grid-dash` | `packages/lattice-grid-dash` | Dash component. |

## The version number is the grid version

A wheel is **`<grid version>.<revision>`**: `1.69.0.0` carries Lattice Grid
`1.69.0`. Nobody chooses it and nobody types it — `tools/bump_grid.py` derives
it from the npm tarball it just downloaded.

The two host wrappers pin `lattice-grid-pandas==<that exact version>`, so all
three move together or pip refuses the install. A **revision** above `0` is for
a packaging fix against an unchanged grid (`--revision 1` -> `1.69.0.1`).

> Before BACKLOG-0001110 these were `0.1.0` and carried a `GRID_VERSION`
> constant typed into Python. It said `1.40.0` while npm had reached `1.69.0`:
> twenty-nine grid releases that never reached a `pip install`. There is now no
> grid version in any Python source at all — `lattice_grid_pandas.GRID_VERSION`
> reads `grid_bundle.json`, which `bump_grid.py` writes from the tarball's own
> `package.json`, and `tests/test_grid_version.py` fails if the constant, the
> banner inside the vendored bundle, the wheel's number and the CDN URL stop
> agreeing.

---

## Route A: automatic (the normal path — nobody does anything)

When the grid publishes a version to npm, its `publish-npm.yml` sends this
repository a `repository_dispatch` of type `grid-released` carrying
`{ "version": "1.69.0" }`. That starts `.github/workflows/publish-pypi.yml`:

1. **bump** — downloads that exact tarball, vendors `lattice-grid.min.js` and
   `lattice-grid.min.css` into the Jupyter wheel, rebuilds the Dash component's
   browser bundle (esbuild compiles the grid *into* it), and rewrites every
   version string in the tree;
2. **test** — installs the three packages the way a user would and runs the
   whole suite, including the real-browser smoke tests, against the new bundle.
   Chromium is installed so those tests *run* rather than skip. **A red test is
   a release that does not happen**;
3. **commit** — one commit on `main`, `Lattice Grid X -> Python wrappers X.0`,
   so the repository and PyPI never disagree about what shipped;
4. **publish** — builds sdists + wheels from that commit, asserts each wheel
   really carries the grid its number promises, and publishes all three over
   PyPI Trusted Publishing (OIDC, no stored token).

Nothing here needs a person. If a grid release did not produce wheels, look at
the Actions tab of this repository first and the grid's `notify-python` job
second (a missing `PYTHON_DISPATCH_TOKEN` there is a notice, not a failure).

### Running it by hand

Actions -> **Publish Python packages to PyPI** -> Run workflow, and give it a
grid version (`1.69.0`). Same job, same tests, same publish. Leave the version
empty to publish the tree exactly as it stands — a packaging-only re-release.

---

## Route B: locally, for a dry run or an emergency

```bash
# 1. vendor the grid and rewrite every version (needs npm on the path)
python tools/bump_grid.py --grid-version 1.69.0 --build-dash

# 2. install the three as a user would, and test against the new bundle
python -m pip install ./packages/lattice-grid-pandas . ./packages/lattice-grid-dash
python -m pip install pytest playwright && python -m playwright install chromium
python -m pytest tests packages/lattice-grid-dash/tests -v

# 3. build artifacts
python -m pip install --upgrade build
rm -rf dist-local && mkdir -p dist-local
for d in packages/lattice-grid-pandas packages/lattice-grid-dash .; do
  ( cd "$d" && python -m build --outdir "$OLDPWD/dist-local" )
done
ls -la dist-local   # 3 wheels + 3 sdists
```

Sanity-check an artifact in a throwaway venv before releasing:

```bash
python -m venv /tmp/verify
/tmp/verify/bin/pip install --find-links dist-local "lattice-grid-jupyter==1.69.0.0"
/tmp/verify/bin/python -c "import lattice_grid_pandas as p; print(p.__version__, p.GRID_VERSION)"
```

Upload with `twine` only if the workflow cannot: **you** supply the PyPI token,
never in a tracked file.

```bash
python -m pip install --upgrade twine
python -m twine check dist-local/*
TWINE_USERNAME=__token__ TWINE_PASSWORD=<your-token> python -m twine upload dist-local/*
```

---

## What the owner set up once

**Here:**
- PyPI Trusted Publisher for each of the three project names, pointing at this
  repo + `publish-pypi.yml` + the matching environment
  (`pypi-pandas` / `pypi` / `pypi-dash`). Trust is bound to the **workflow
  filename**, which is why the bump lives inside `publish-pypi.yml` rather than
  in a workflow of its own.
- Those three GitHub Environments.

**In the grid repository (`tocloco/lattice`):**
- The `PYTHON_DISPATCH_TOKEN` secret: a fine-grained PAT issued by `toclocoinc`,
  scoped to `toclocoinc/lattice-grid-python` alone, **Contents: read and write**
  (the permission `repository_dispatch` lives under). Without it the grid still
  publishes and simply prints a notice that the wrappers were not told.

`bump` pushes one commit to `main` with the default `GITHUB_TOKEN`, so `main`
must not be protected against it (or the push needs the same PAT).
