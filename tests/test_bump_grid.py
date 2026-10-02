"""Unit tests for ``tools/bump_grid.py`` that need no network and no real delay.

``wait_for_npm_version`` exists because the automatic bump runs about a minute
after the grid's own ``npm publish``, and npm's registry takes 7 to 12 minutes
to start serving a version it has just accepted -- packing that early is a 404
and the whole release fails silently (BACKLOG-0001441, grid 1.71.0, run
35915717001). These tests fake the registry and the clock so the loop's
behaviour is asserted without spending twenty real minutes on it.
"""

import importlib.util
import pathlib

import pytest

_TOOLS_DIR = pathlib.Path(__file__).resolve().parent.parent / "tools"
_spec = importlib.util.spec_from_file_location("bump_grid", _TOOLS_DIR / "bump_grid.py")
bump_grid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bump_grid)


class _FakeResult:
    """Stands in for a ``subprocess.CompletedProcess``."""

    def __init__(self, returncode: int, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


def test_wait_for_npm_version_returns_once_the_registry_answers():
    """The third `npm view` call reports the version; the loop stops there."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) < 3:
            return _FakeResult(returncode=1, stdout="")  # npm ERR! 404, not published yet
        return _FakeResult(returncode=0, stdout="1.71.0\n")

    sleeps = []

    bump_grid.wait_for_npm_version(
        "1.71.0",
        interval=30.0,
        timeout=20 * 60.0,
        run=fake_run,
        sleep=sleeps.append,
        log=lambda message: None,
    )

    assert len(calls) == 3
    assert calls[0] == ["npm", "view", "@toclocoinc/lattice-grid@1.71.0", "version"]
    # Slept between attempt 1->2 and 2->3, never after the answer arrived.
    assert sleeps == [30.0, 30.0]


def test_wait_for_npm_version_logs_every_attempt():
    """Every poll is logged, so a hung wait is visible in the CI log, not silent."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) < 3:
            return _FakeResult(returncode=1, stdout="")
        return _FakeResult(returncode=0, stdout="1.71.0\n")

    logged = []

    bump_grid.wait_for_npm_version(
        "1.71.0",
        interval=30.0,
        timeout=20 * 60.0,
        run=fake_run,
        sleep=lambda seconds: None,
        log=logged.append,
    )

    assert len(logged) == 4  # 3 polling attempts + the success line
    assert all("1.71.0" in line for line in logged)
    assert "attempt 1/40" in logged[0]
    assert "attempt 3/40" in logged[2]
    assert "live on the registry" in logged[-1]


def test_wait_for_npm_version_gives_up_after_the_deadline():
    """A version that never appears fails, naming the wait and the version."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _FakeResult(returncode=1, stdout="")  # never answers

    sleeps = []

    with pytest.raises(SystemExit) as excinfo:
        bump_grid.wait_for_npm_version(
            "9.9.9",
            interval=30.0,
            timeout=120.0,  # 4 attempts at a 30s interval
            run=fake_run,
            sleep=sleeps.append,
            log=lambda message: None,
        )

    assert len(calls) == 4
    assert sleeps == [30.0, 30.0, 30.0]  # never sleeps after the last attempt
    message = str(excinfo.value)
    assert "9.9.9" in message
    assert "waiting" in message
    assert "2 minutes" in message


def test_wait_for_npm_version_rejects_a_mismatched_answer():
    """`npm view` succeeding on a DIFFERENT version (a stale cache) must not pass."""

    def fake_run(cmd, **kwargs):
        return _FakeResult(returncode=0, stdout="1.70.0\n")  # the previous release

    with pytest.raises(SystemExit):
        bump_grid.wait_for_npm_version(
            "1.71.0",
            interval=30.0,
            timeout=60.0,
            run=fake_run,
            sleep=lambda seconds: None,
            log=lambda message: None,
        )


# --- the chart and router modules, and the name tables -------------------------


def _fake_package(root: pathlib.Path, names) -> pathlib.Path:
    """A tarball-shaped directory whose modules end the way the grid's do."""
    package = root / "package"
    (package / "modules").mkdir(parents=True)
    for name in names:
        (package / "modules" / f"{name}.min.js").write_text(
            f"/*! Lattice Grid 9.9.9, {name} module */\n(function(){{}})();\n"
            f"//# sourceMappingURL={name}.min.js.map\n",
            encoding="utf-8",
        )
    return package


def test_vendor_modules_copies_each_module_without_its_source_map(tmp_path, monkeypatch):
    """Every listed module lands where the widget reads it, map reference stripped."""
    monkeypatch.setattr(bump_grid, "ROOT", tmp_path)
    package = _fake_package(tmp_path / "tgz", ["charts", "chart-fan", "data-router"])
    dests = {
        "charts": tmp_path / "static/charts/charts.min.js",
        "chart-fan": tmp_path / "static/charts/chart-fan.min.js",
        "data-router": tmp_path / "static/router/data-router.min.js",
    }
    written = bump_grid.vendor_modules(package, dests)
    assert written == list(dests.values())
    for name, dest in dests.items():
        text = dest.read_text(encoding="utf-8")
        assert text.startswith(f"/*! Lattice Grid 9.9.9, {name} module */")
        assert "sourceMappingURL" not in text
        assert text.endswith("(function(){})();\n\n")


def test_vendor_modules_stops_on_a_module_the_tarball_lacks(tmp_path, monkeypatch):
    """A missing module is a failed bump naming the file, never a silent skip."""
    monkeypatch.setattr(bump_grid, "ROOT", tmp_path)
    package = _fake_package(tmp_path / "tgz", ["charts"])
    with pytest.raises(SystemExit, match="modules/chart-roc.min.js"):
        bump_grid.vendor_modules(package, {
            "charts": tmp_path / "charts.min.js",
            "chart-roc": tmp_path / "chart-roc.min.js",
        })


def test_the_vendored_chart_modules_are_the_widgets_extension_types():
    """bump_grid vendors exactly the chart type modules ``LatticeChart`` loads."""
    chart = pytest.importorskip("lattice_grid_jupyter._chart")
    assert set(bump_grid.CHART_TYPE_MODULES) == {f"chart-{t}" for t in chart.EXTENSION_TYPES}
    assert set(bump_grid.MODULES) == {"charts", "data-router", *bump_grid.CHART_TYPE_MODULES}
    assert bump_grid.MODULES["data-router"].parent.name == "router"


def test_main_vendors_the_modules_and_regenerates_the_tables(tmp_path, monkeypatch):
    """The bump itself runs both steps on the tarball it fetched -- so an automatic
    release carries the chart and router modules and the tables at the new grid."""
    package = tmp_path / "package"
    package.mkdir()
    (package / "package.json").write_text('{"version": "9.9.9"}', encoding="utf-8")
    calls = []
    monkeypatch.setattr(bump_grid, "fetch_tarball", lambda version, into: package)
    for step in ("vendor_bundle", "vendor_modules", "regenerate_name_tables"):
        monkeypatch.setattr(bump_grid, step, lambda pkg, _s=step: calls.append((_s, pkg)))
    monkeypatch.setattr(bump_grid, "write_grid_bundle_json", lambda v: calls.append(("json", v)))
    monkeypatch.setattr(bump_grid, "rewrite_versions", lambda g, w: calls.append(("versions", g, w)))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    assert bump_grid.main(["--grid-version", "9.9.9", "--revision", "1"]) == 0
    assert ("vendor_modules", package) in calls
    assert ("regenerate_name_tables", package) in calls
    assert ("versions", "9.9.9", "9.9.9.1") in calls


def test_regenerate_name_tables_reads_the_tarball_declarations(tmp_path, monkeypatch):
    """Both generators run against the tarball's own ``lattice-grid.d.ts``."""
    tools = tmp_path / "tools"
    tools.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    for tool in ("gen_option_names", "gen_chart_names"):
        (tools / f"{tool}.py").write_text(
            "import pathlib\n"
            f"OUT = pathlib.Path({str(out / tool)!r})\n"
            "def main(dts):\n"
            "    OUT.write_text(pathlib.Path(dts).read_text())\n"
            "    return 0\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(bump_grid, "ROOT", tmp_path)
    package = tmp_path / "package"
    package.mkdir()
    (package / "lattice-grid.d.ts").write_text("export interface X {}\n", encoding="utf-8")
    bump_grid.regenerate_name_tables(package)
    for tool in ("gen_option_names", "gen_chart_names"):
        assert (out / tool).read_text() == "export interface X {}\n"
