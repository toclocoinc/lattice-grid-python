"""Run one router declaration through the vendored browser engine (node) and the kernel."""

import json
import pathlib
import shutil
import subprocess

import lattice_grid_jupyter
from lattice_grid_jupyter._router_engine import shape_route

_HERE = pathlib.Path(__file__).resolve().parent
_BUNDLE = pathlib.Path(lattice_grid_jupyter.__file__).parent / "static" / "router" / "data-router.min.js"
NODE = shutil.which("node")


def browser_engine(router, route):
    """The shaped columnar rows the vendored data-router bundle produces under node."""
    payload = json.loads(json.dumps(router._payload(route)))   # exactly what travels to the front end
    done = subprocess.run([NODE, str(_HERE / "_router_node.js")], capture_output=True, text=True, timeout=60,
                          input=json.dumps({"bundle": str(_BUNDLE), "payload": payload}))
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def kernel_engine(router, route):
    return shape_route(router._payload(route))


def both(router, route):
    return browser_engine(router, route), kernel_engine(router, route)
