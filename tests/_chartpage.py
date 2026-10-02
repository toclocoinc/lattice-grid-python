"""Real-browser harness for several widgets in ONE page (a grid and its charts).

Each widget gets its own mock model and its own front end (``widget.js`` for a
grid, ``chart.js`` for a chart), exactly as a notebook renders them side by side
in one JS realm. ``window.__sets[i]`` logs what widget ``i`` sent to its model;
``window.__rendered`` is true once every widget has rendered.
"""

import json
import pathlib

import lattice_grid_jupyter

_STATIC = pathlib.Path(lattice_grid_jupyter.__file__).parent / "static"

_TMPL = """<!doctype html><html><head><meta charset=utf-8></head><body>
<div id=root></div>
<script>
window.__sets = []; window.__models = []; window.__sent = [];
function mkModel(i, state) {
  const L = {}; window.__sets[i] = [];
  const m = {
    get: k => state[k],
    set: (k, v) => { state[k] = v; window.__sets[i].push([k, JSON.parse(JSON.stringify(v === undefined ? null : v))]); },
    save_changes: () => {},
    send: (content) => {
      window.__sent.push([i, content && content.method]);
      if (!window.__pysend) return;
      window.__pysend(i, content).then((reply) => {
        if (reply) (L['msg:custom']||[]).slice().forEach(f => f(reply, []));
      });
    },
    on: (e, cb) => { (L[e]=L[e]||[]).push(cb); },
    off: (e, cb) => { L[e] = (L[e]||[]).filter(f=>f!==cb); },
  };
  window.__models[i] = m;
  window.__pyset = window.__pyset || ((j, k, v) => { window.__states[j][k] = v; (window.__listeners[j]['change:'+k]||[]).forEach(f=>f()); });
  window.__states = window.__states || []; window.__listeners = window.__listeners || [];
  window.__states[i] = state; window.__listeners[i] = L;
  return m;
}
</script>
__SCRIPTS__
</body></html>"""

_ONE = """<div id=root__I__></div>
<script type=module>
__WIDGET__
;(async () => {
  window.__done = window.__done || 0;
  try {
    await window.__w__I__.render({ model: mkModel(__I__, __STATE__), el: document.getElementById('root__I__') });
  } catch (e) { window.__error = String(e && e.stack || e); }
  window.__done += 1;
  if (window.__done === __N__) window.__rendered = true;
})();
</script>"""


def page_html(widgets) -> str:
    parts = []
    for i, w in enumerate(widgets):
        js_name = "chart.js" if type(w).__name__ == "LatticeChart" else "widget.js"
        js = (_STATIC / js_name).read_text().replace("export default", f"window.__w{i} =")
        state = {k: v for k, v in w.get_state().items() if not k.startswith("_esm") and k != "_css"}
        state.setdefault("_edit", None)
        parts.append(_ONE.replace("__WIDGET__", js).replace("__STATE__", json.dumps(state))
                     .replace("__I__", str(i)).replace("__N__", str(len(widgets))))
    return _TMPL.replace("__SCRIPTS__", "\n".join(parts))


def bind_python(page, widgets):
    """Route widget ``i``'s comm messages to the real Python widget ``i`` (call before set_content)."""
    def handle(i, content):
        w = widgets[i]
        if isinstance(content, dict) and content.get("type") == "lg:req" and hasattr(w, "_answer"):
            return w._answer(content)
        return None
    page.expose_function("__pysend", handle)


def sets(page, i, key=None):
    got = page.evaluate(f"window.__sets[{i}]")
    return [v for k, v in got if key is None or k == key] if key else got
