"""Make the shared test helpers (``_hosts``, ``_page``, ``_chartpage``, ``_fixture``) importable.

``pyproject.toml`` runs pytest with ``--import-mode=importlib`` (so this suite's
``test_smoke_browser.py`` and the Dash package's can share a basename), and that
mode does not put a test file's directory on ``sys.path``. The helper modules
live beside the tests, so this directory is added here, once.
"""

import pathlib
import sys

_HERE = str(pathlib.Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
