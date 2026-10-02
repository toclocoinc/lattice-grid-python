/*
 * lattice-grid-jupyter -- AnyWidget front end.
 *
 * Reconstructs a columnar payload into row objects, renders a Lattice grid via
 * createGrid, and wires the bidirectional bridge:
 *   grid cell:changed  -> model._edit          (edit -> DataFrame, Python side)
 *   model._data_version -> grid.rows.load(...)  (Python set_data -> grid)
 *   model._row_op       -> grid.rows.apply(...) (Python append/delete -> grid)
 *
 * The grid bundle loads one of two ways, chosen by model._grid_source:
 *   "cdn"    -> dynamic import() of the published ESM build from jsDelivr.
 *   "vendor" -> the UMD bundle text carried in model._grid_js is injected as a
 *               <script> (publishes window.LatticeGrid); CSS from _grid_css.
 *               Fully offline: no network at render time.
 */

const CDN_BASE = (v) => `https://cdn.jsdelivr.net/npm/@toclocoinc/lattice-grid@${v}`;

function reconstructRows(model) {
  const columnar = model.get("_columnar") || {};
  const keys = columnar["__row_id__"] || [];
  const fields = Object.keys(columnar).filter((f) => f !== "__row_id__");
  const n = keys.length;
  const rows = new Array(n);
  for (let i = 0; i < n; i++) {
    const rec = { __row_id__: keys[i] };
    for (let c = 0; c < fields.length; c++) {
      rec[fields[c]] = columnar[fields[c]][i];
    }
    rows[i] = rec;
  }
  return rows;
}

async function loadGrid(model) {
  const source = model.get("_grid_source") || "cdn";

  if (source === "vendor") {
    // Inject CSS from the carried text.
    if (!document.querySelector("style[data-lattice-css]")) {
      const style = document.createElement("style");
      style.setAttribute("data-lattice-css", "1");
      style.textContent = model.get("_grid_css") || "";
      document.head.appendChild(style);
    }
    // Inject the UMD bundle once -> window.LatticeGrid.
    if (!window.LatticeGrid) {
      const js = model.get("_grid_js") || "";
      if (!js) throw new Error("vendor mode but _grid_js is empty");
      const script = document.createElement("script");
      script.textContent = js;
      document.head.appendChild(script);
    }
    if (!window.LatticeGrid || typeof window.LatticeGrid.createGrid !== "function") {
      throw new Error("vendored bundle did not expose window.LatticeGrid.createGrid");
    }
    return window.LatticeGrid;
  }

  // CDN mode: dynamic ESM import, deduped on the window. The version comes from
  // the Python side (which reads it from the vendored tarball) and from nowhere
  // else: a literal fallback here is a second place to be wrong, and it was --
  // it still said 1.40.0 at grid 1.69.0 (BACKLOG-0001110).
  const version = model.get("_grid_version");
  if (!version) throw new Error("cdn mode but _grid_version is empty");
  const base = CDN_BASE(version);
  if (!document.querySelector("link[data-lattice-css]")) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = base + "/lattice-grid.min.css";
    link.setAttribute("data-lattice-css", "1");
    document.head.appendChild(link);
  }
  if (!window.__latticeGridMod) {
    window.__latticeGridMod = import(base + "/lattice-grid.esm.min.js");
  }
  return window.__latticeGridMod;
}

async function render({ model, el }) {
  const mod = await loadGrid(model);
  const createGrid = mod.createGrid;

  const host = document.createElement("div");
  host.style.height = (model.get("height") || 360) + "px";
  host.style.width = "100%";
  el.appendChild(host);

  const licence = model.get("licence");
  let appliedOptions = model.get("_grid_options") || {};
  const initialState = model.get("state");
  const grid = createGrid(host, {
    selection: "multiple",
    edit: true,
    ...appliedOptions,
    // Python owns the data, the row key and the column set.
    columns: model.get("_columns"),
    rows: reconstructRows(model),
    rowKey: model.get("_row_key") || "__row_id__",
    ...(initialState && Object.keys(initialState).length ? { state: initialState } : {}),
    ...(licence ? { licence } : {}),
  });
  host.__latticeGrid = grid; // handle for tests and notebook-side debugging

  // Surface the resolved licence state for debugging / the smoke test.
  try {
    if (grid.licence && typeof grid.licence.state === "function") {
      model.set("_licence_state", String(grid.licence.state()));
      model.save_changes();
    }
  } catch (e) { /* non-fatal */ }

  // --- grid edit -> DataFrame -------------------------------------------------
  const onCellChanged = (e) => {
    model.set("_edit", {
      key: String(e.key),
      colId: e.colId,
      value: e.value,
      old: e.oldValue,
      ts: Date.now(),
    });
    model.save_changes();
  };
  grid.on("cell:changed", onCellChanged);

  // --- Python set_data(df) -> full reload ------------------------------------
  const onData = () => {
    const rows = reconstructRows(model);
    if (typeof grid.rows.load === "function") {
      // load() keeps the column layout; set_data assumes the same schema.
      grid.rows.load(rows);
    }
  };
  model.on("change:_data_version", onData);

  // --- Python append_rows / delete_rows -> incremental apply -----------------
  const onRowOp = () => {
    const op = model.get("_row_op");
    if (!op || !op.op) return;
    if (typeof grid.rows.apply !== "function") return;
    if (op.op === "add") {
      grid.rows.apply({ add: op.rows || [], ...(op.at != null ? { at: op.at } : {}) });
    } else if (op.op === "remove") {
      grid.rows.apply({ remove: op.keys || [] });
    } else if (op.op === "update") {
      grid.rows.apply({ update: op.rows || [] });
    }
  };
  model.on("change:_row_op", onRowOp);

  // --- grid -> Python: selection, filtered view, state (debounced) ----------
  // Only KEYS cross the comm: the frame already lives in Python, which slices it.
  const DEBOUNCE_MS = 120;
  const debounced = (fn) => {
    let t = null;
    const run = () => { t = null; fn(); };
    const call = () => { if (t !== null) clearTimeout(t); t = setTimeout(run, DEBOUNCE_MS); };
    call.cancel = () => { if (t !== null) clearTimeout(t); t = null; };
    return call;
  };
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  const sendSelection = () => {
    const keys = grid.selection.keys().map(String);
    if (same(keys, model.get("_selected_keys"))) return;
    model.set("_selected_keys", keys);
    model.save_changes();
  };
  const sendView = () => {
    const keys = [];
    grid.rows.forEach((row) => { if (row.data != null) keys.push(String(row.key)); });
    if (same(keys, model.get("_view_keys"))) return;
    model.set("_view_keys", keys);
    model.save_changes();
  };
  const STATE_SECTIONS = ["columns", "columnOrder", "columnGroups", "filters", "where",
    "quick", "quickMode", "sort", "group", "pivot"];
  let lastState = null;
  const sendState = () => {
    const full = grid.state.get();
    const plain = { version: full.version };
    for (const k of STATE_SECTIONS) if (full[k] !== undefined) plain[k] = full[k];
    const json = JSON.stringify(plain);
    if (json === lastState) return;
    lastState = json;
    model.set("state", JSON.parse(json));
    model.save_changes();
  };
  const dSelection = debounced(sendSelection);
  const dView = debounced(sendView);
  const dState = debounced(sendState);
  grid.on("selection:changed", dSelection);
  grid.on("model:changed", dView);
  grid.on("state:changed", dState);
  // first report once the grid has painted, so Python's view/state are real
  grid.on("ready", () => { sendView(); sendState(); sendSelection(); });

  // --- Python widget.state = {...} -> restore --------------------------------
  const onState = () => {
    const st = model.get("state");
    if (!st || !Object.keys(st).length) return;
    if (JSON.stringify(st) === lastState) return; // our own echo
    lastState = JSON.stringify(st);
    grid.state.apply(st);
  };
  model.on("change:state", onState);

  // --- grid warnings -> Python warnings.warn ---------------------------------
  const sentWarnings = new Set();
  const reportWarnings = () => {
    let list = [];
    try { list = grid.diagnostics.warnings(); } catch (e) { return; }
    const fresh = [];
    for (const w of list) {
      if (w.source !== "reported" || sentWarnings.has(w.id)) continue;
      if (!/^(config|column|columns)[.:]/.test(w.id)) continue;
      sentWarnings.add(w.id);
      fresh.push({ id: w.id, message: w.message });
    }
    if (fresh.length) {
      model.set("_grid_warnings", fresh);
      model.save_changes();
    }
  };
  setTimeout(reportWarnings, 0);

  // --- Python widget.options = {...} -> live update, no data re-sent ---------
  const onOptions = () => {
    const next = model.get("_grid_options") || {};
    const values = {};
    for (const k of Object.keys(next)) if (!same(next[k], appliedOptions[k])) values[k] = next[k];
    for (const k of Object.keys(appliedOptions)) if (!(k in next)) values[k] = undefined;
    appliedOptions = next;
    if (!Object.keys(values).length) return;
    if (typeof grid.setAll === "function") grid.setAll(values);
    else for (const k of Object.keys(values)) grid.set(k, values[k]);
    setTimeout(reportWarnings, 0);
  };
  model.on("change:_grid_options", onOptions);

  return () => {
    model.off("change:_grid_options", onOptions);
    model.off("change:state", onState);
    dSelection.cancel(); dView.cancel(); dState.cancel();
    model.off("change:_data_version", onData);
    model.off("change:_row_op", onRowOp);
    grid.off && grid.off("cell:changed", onCellChanged);
    grid.destroy && grid.destroy();
  };
}

export default { render };
