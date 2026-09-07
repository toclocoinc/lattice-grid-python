# Releasing the Lattice Grid Python packages (1.41)

This repo ships three separately-installable packages, released together at
**0.1.0**:

| Package | Path | Purpose |
| --- | --- | --- |
| `lattice-grid-pandas` | `packages/lattice-grid-pandas` | Shared, framework-agnostic pandas <-> grid serialization + grid/licence plumbing. |
| `lattice-grid-jupyter` | `.` (repo root) | anywidget notebook widget. Depends on `lattice-grid-pandas==0.1.0`. |
| `lattice-grid-dash` | `packages/lattice-grid-dash` | Dash component. Depends on `lattice-grid-pandas==0.1.0`. |

Both wrappers pin the shared package to **`==0.1.0`**, so release all three
together and bump their versions in lockstep.

> `lattice-grid-dash` commits its built browser bundle
> (`lattice_grid_dash/lattice_grid_dash.min.js`) and generated Python
> (`lattice_grid_dash/LatticeGrid.py`, `metadata.json`). **Rebuild them before
> tagging** if the component source changed:
> ```bash
> cd packages/lattice-grid-dash && npm install && npm run build
> ```

## Build artifacts locally (both routes use this)

```bash
python -m pip install --upgrade build
rm -rf dist-1.41 && mkdir -p dist-1.41
for d in packages/lattice-grid-pandas packages/lattice-grid-dash .; do
  ( cd "$d" && python -m build --outdir "$OLDPWD/dist-1.41" )
done
ls -la dist-1.41   # 3 wheels + 3 sdists
```

Sanity-check an artifact in a throwaway venv before releasing:

```bash
python -m venv /tmp/verify && /tmp/verify/bin/pip install \
  --find-links dist-1.41 "lattice-grid-dash==0.1.0"
/tmp/verify/bin/python -c "import lattice_grid_dash; print(lattice_grid_dash.__version__)"
```

---

## Route A (recommended): PyPI Trusted Publishing via GitHub Actions (OIDC)

No API tokens are stored anywhere. GitHub Actions mints a short-lived OIDC token
that PyPI trusts, scoped to a specific repo + workflow + environment.

**Already in this repo:** `.github/workflows/publish-pypi.yml` — on a pushed tag
`v*` it builds all three packages and publishes each with
`pypa/gh-action-pypi-publish` (`id-token: write`, no secrets).

### One-time owner setup

1. **Choose/create a GitHub repository** for these packages and push this tree to
   it (including `.github/workflows/publish-pypi.yml`). Trusted publishing is tied
   to a specific `owner/repo` + workflow filename.
   *(This has NOT been done for you — pick the org/repo yourself.)*

2. **(Recommended) create a GitHub Environment** named **`pypi`** on that repo
   (Settings -> Environments -> New environment). The workflow references it; add
   required reviewers here if you want a manual approval gate before each publish.

3. **On PyPI, add a Trusted Publisher for each of the three projects.** PyPI
   supports *pending* publishers, so you can do this **before** the projects
   exist — the first successful publish then creates the project. For **each** of
   `lattice-grid-pandas`, `lattice-grid-jupyter`, `lattice-grid-dash`:
   - Go to <https://pypi.org/manage/account/publishing/> ("Add a new pending
     publisher"), or, if the project already exists, its
     *Manage -> Publishing* page.
   - Fill in:
     - **PyPI Project Name**: the package name (e.g. `lattice-grid-dash`)
     - **Owner**: your GitHub org/user
     - **Repository name**: the repo you chose in step 1
     - **Workflow name**: `publish-pypi.yml`
     - **Environment name**: `pypi`

### Cutting a release

```bash
git tag v0.1.0
git push origin v0.1.0        # triggers the workflow -> builds + publishes all 3
```

To dry-run against TestPyPI first, add pending publishers on
<https://test.pypi.org> and set `repository-url: https://test.pypi.org/legacy/`
in the publish step (a commented hint is in the workflow).

---

## Route B: Manual upload with `twine`

For a one-off publish from a trusted machine. **You** supply the PyPI token; never
hardcode it in a file or commit it.

```bash
# 1. build (see "Build artifacts locally" above) -> dist-1.41/

# 2. check the metadata renders
python -m pip install --upgrade twine
python -m twine check dist-1.41/*

# 3. upload. Provide your token interactively / via your own env; do NOT paste it
#    into any tracked file. Username is literally __token__.
python -m twine upload dist-1.41/*
```

- When prompted, username = `__token__`, password = your PyPI API token
  (`pypi-...`).
- Or, in your own shell/session only:
  `TWINE_USERNAME=__token__ TWINE_PASSWORD=<your-token> python -m twine upload dist-1.41/*`
  (e.g. in a notebook cell: `! python -m twine upload dist-1.41/*` and enter the
  token at the prompt).
- TestPyPI dry-run: `python -m twine upload --repository testpypi dist-1.41/*`.

---

## Minimal owner checklist

**Route A (OIDC):**
- Create/choose a GitHub repo; push this tree (with the workflow).
- Create the `pypi` GitHub Environment (recommended).
- Add a PyPI trusted/pending publisher for each of the 3 project names
  (repo + `publish-pypi.yml` + `pypi`).
- `git tag v0.1.0 && git push origin v0.1.0`.

**Route B (manual):**
- `python -m build` each package into `dist-1.41/`.
- `python -m twine check dist-1.41/*`.
- `python -m twine upload dist-1.41/*` with your own `__token__` (never stored).
