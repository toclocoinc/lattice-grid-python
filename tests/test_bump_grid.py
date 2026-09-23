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
