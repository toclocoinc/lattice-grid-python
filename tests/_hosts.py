"""Shared helpers: a real Jupyter server (Lab 4 or Notebook 7) + headless Chrome."""

from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def launch_browser(p):
    for channel in ("chrome", "chromium", "msedge"):
        try:
            return p.chromium.launch(channel=channel, headless=True, args=["--no-sandbox"])
        except Exception:
            continue
    try:
        return p.chromium.launch(headless=True, args=["--no-sandbox"])
    except Exception:
        return None


def notebook_json(cells: list[str]) -> dict:
    return {
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": src}
            for src in cells
        ],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                    "language": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


class JupyterServer:
    """A throw-away Jupyter server we started (and only we kill, by Popen handle)."""

    def __init__(self, flavour: str, cells: list[str]):
        assert flavour in ("lab", "notebook")
        self.flavour = flavour
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix=f"lgp-{flavour}-"))
        (self.dir / "nb.ipynb").write_text(json.dumps(notebook_json(cells)))
        self.port = free_port()
        self.token = "lgp" + str(os.getpid())
        env = dict(os.environ,
                   JUPYTER_CONFIG_DIR=str(self.dir / "cfg"),
                   JUPYTER_RUNTIME_DIR=str(self.dir / "rt"),
                   JUPYTER_DATA_DIR=str(self.dir / "data"),
                   JUPYTER_PLATFORM_DIRS="1")
        # kernels come from this interpreter's prefix, not the throw-away data dir
        env["JUPYTER_PATH"] = str(pathlib.Path(sys.prefix) / "share" / "jupyter")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", flavour if flavour == "notebook" else "jupyterlab",
             "--no-browser", f"--port={self.port}", "--ip=127.0.0.1",
             f"--ServerApp.token={self.token}", "--ServerApp.password=",
             "--expose-app-in-browser", f"--notebook-dir={self.dir}",
             "--ServerApp.disable_check_xsrf=True"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api?token={self.token}", timeout=2)
                return
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"{flavour} server exited early")
                time.sleep(0.5)
        self.close()
        raise RuntimeError(f"{flavour} server did not come up")

    @property
    def url(self) -> str:
        base = f"http://127.0.0.1:{self.port}"
        if self.flavour == "lab":
            return f"{base}/lab/tree/nb.ipynb?token={self.token}&reset"
        return f"{base}/notebooks/nb.ipynb?token={self.token}"

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
