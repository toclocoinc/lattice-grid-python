"""The version the wrappers CLAIM is the version they SHIP.

These three wheels vendor a grid bundle rather than resolving one at install
time, so "which grid is this?" is answered in four places at once: the constant
Python reports, the banner inside the vendored JavaScript, the wheel's own
version number, and the URL the CDN path builds. They drifted apart and stayed
apart -- ``GRID_VERSION`` read ``1.40.0`` through twenty-nine grid releases,
while the CDN URL it built pointed at a bundle nobody had tested against
(BACKLOG-0001110). Nothing in the suite noticed, because nothing compared the
constant to the bytes.

This compares them. ``tools/bump_grid.py`` is the only writer of any of them,
and these tests fail the moment one of its edits is missed.
"""

import json
import pathlib
import re

import pytest

import lattice_grid_pandas
import lattice_grid_jupyter

#: `/*! Lattice Grid 1.69.0, core + DOM renderer` -- the banner every build stamps.
_STAMP = re.compile(r"Lattice Grid (\d+\.\d+\.\d+)")

_JUPYTER_BUNDLE = (
    pathlib.Path(lattice_grid_jupyter.__file__).parent / "static" / "lattice-grid.min.js"
)


def _stamps(path: pathlib.Path) -> set:
    """Every distinct grid version stamped inside a built bundle."""
    return set(_STAMP.findall(path.read_text(encoding="utf-8", errors="replace")))


def test_grid_version_is_read_from_the_vendored_record_not_typed():
    """GRID_VERSION comes out of the generated file, so no source can hold a stale one."""
    record = (
        pathlib.Path(lattice_grid_pandas.__file__).parent / "grid_bundle.json"
    )
    assert record.exists(), "grid_bundle.json did not ship in the wheel"
    data = json.loads(record.read_text(encoding="utf-8"))
    assert data["grid_version"] == lattice_grid_pandas.GRID_VERSION
    assert data["npm_package"] == "@toclocoinc/lattice-grid"


def test_the_vendored_bundle_is_the_grid_version_claimed():
    """The bytes in the wheel carry exactly the version the constant reports."""
    assert _JUPYTER_BUNDLE.exists(), f"no vendored bundle at {_JUPYTER_BUNDLE}"
    assert _stamps(_JUPYTER_BUNDLE) == {lattice_grid_pandas.GRID_VERSION}


def test_both_host_wrappers_report_one_grid_version():
    """The Jupyter widget re-exports the shared constant rather than keeping a copy."""
    assert lattice_grid_jupyter.GRID_VERSION is lattice_grid_pandas.GRID_VERSION


@pytest.mark.parametrize(
    "package", [lattice_grid_pandas, lattice_grid_jupyter], ids=["pandas", "jupyter"]
)
def test_the_wheel_version_names_the_grid_it_carries(package):
    """A wheel is `<grid>.<revision>`, so its number alone says which grid is inside."""
    assert package.__version__.startswith(lattice_grid_pandas.GRID_VERSION + ".")
    revision = package.__version__[len(lattice_grid_pandas.GRID_VERSION) + 1 :]
    assert revision.isdigit(), f"{package.__name__} {package.__version__} has no revision"


def test_the_cdn_urls_point_at_the_version_that_was_tested():
    """CDN delivery quotes the same version vendor delivery ships -- not a default.

    Both entry points are asserted, because they carry the default separately:
    a mutation test put a literal back on ``CDN_BASE`` and ``cdn_urls()`` alone
    stayed green, since it passes its own default down.
    """
    version = lattice_grid_pandas.GRID_VERSION
    assert lattice_grid_pandas.CDN_BASE().endswith(f"@toclocoinc/lattice-grid@{version}")
    urls = lattice_grid_pandas.cdn_urls()
    assert set(urls) == {"core", "react", "css"}
    for name, url in urls.items():
        assert f"@toclocoinc/lattice-grid@{version}/" in url, f"{name}: {url}"


def test_the_dash_component_carries_the_same_grid():
    """The Dash bundle compiles the grid in, so it must be rebuilt by the same bump."""
    dash = pytest.importorskip("lattice_grid_dash", reason="lattice-grid-dash not installed")
    bundle = pathlib.Path(dash.__file__).parent / "lattice_grid_dash.min.js"
    assert bundle.exists(), f"no built Dash bundle at {bundle}"
    assert _stamps(bundle) == {lattice_grid_pandas.GRID_VERSION}
    assert dash.__version__ == lattice_grid_pandas.__version__


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
