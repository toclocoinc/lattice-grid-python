"""Real-browser harness: the widget's actual front end in Chromium with a mock model.

The mock model carries EVERY synced trait of a real widget (``get_state()``), logs
each ``set`` the front end makes (``window.__sets``) and lets a test play the part
of the kernel (``window.__pyset(key, value)`` fires ``change:<key>``).
"""

import json
import pathlib

import lattice_grid_jupyter

_STATIC = pathlib.Path(lattice_grid_jupyter.__file__).parent / "static"

_TMPL = """<!doctype html><html><head><meta charset=utf-8></head><body>
<div id=root></div>
<script>
const state = __STATE__; const L = {}; window.__sets = [];
window.__model = {
  get: k => state[k],
  set: (k, v) => { state[k] = v; window.__sets.push([k, JSON.parse(JSON.stringify(v === undefined ? null : v))]); },
  save_changes: () => {},
  on: (e, cb) => { (L[e]=L[e]||[]).push(cb); },
  off: (e, cb) => { L[e] = (L[e]||[]).filter(f=>f!==cb); },
};
window.__pyset = (k, v) => { state[k] = v; (L['change:'+k]||[]).forEach(f=>f()); };
</script>
<script type=module>
__WIDGET__
;(async () => {
  await window.__widget.render({ model: window.__model, el: document.getElementById('root') });
  window.__rendered = true;
})().catch(e => { window.__error = String(e && e.stack || e); });
</script>
</body></html>"""


def page_html(widget) -> str:
    widget_js = (_STATIC / "widget.js").read_text().replace("export default", "window.__widget =")
    state = {k: v for k, v in widget.get_state().items() if not k.startswith("_esm") and k != "_css"}
    state.setdefault("_edit", None)
    return _TMPL.replace("__STATE__", json.dumps(state)).replace("__WIDGET__", widget_js)


def sets(page, key=None):
    """What the front end has sent to the model so far (optionally one key)."""
    got = page.evaluate("window.__sets")
    return [v for k, v in got if key is None or k == key] if key else got
