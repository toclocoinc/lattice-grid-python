/*
 * lattice-grid-jupyter -- AnyWidget front end.
 *
 * Reconstructs a columnar payload into row objects, renders a Lattice grid via
 * createGrid, and wires the bidirectional bridge:
 *   grid cell:changed  -> model._edit          (edit -> DataFrame, Python side)
 *   model._data_version -> grid.rows.load(...)  (Python set_data -> grid)
 *   model._row_op       -> grid.rows.apply(...) (Python append/delete -> grid)
 *
 * Large frames (card 1618): when model._mode is "windowed" the browser holds
 * no rows. The grid gets a pushdown source whose adapter forwards each query
 * (a row window, group-by aggregates, distinct values, whole-set statistics,
 * header-histogram counts, a column profile) to Python over the widget comm as
 * {type: "lg:req", id, method, payload}; Python answers {type: "lg:res", id,
 * ok, result|error}. The correlation id pairs each answer with its request, so
 * answers may arrive in any order and an abandoned request is simply dropped.
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

function rowsFromColumnar(columnar) {
  const keys = (columnar && columnar["__row_id__"]) || [];
  const fields = Object.keys(columnar || {}).filter((f) => f !== "__row_id__");
  const rows = new Array(keys.length);
  for (let i = 0; i < keys.length; i++) {
    const rec = { __row_id__: keys[i] };
    for (let c = 0; c < fields.length; c++) rec[fields[c]] = columnar[fields[c]][i];
    rows[i] = rec;
  }
  return rows;
}

/**
 * The comm round trip: request/response paired by a correlation id.
 * An aborted request rejects at once with the signal's reason; its late answer
 * (Python cannot be interrupted mid-computation) finds no pending entry and is
 * dropped.
 */
function createCommClient(model) {
  let seq = 0;
  const pending = new Map();
  const onMsg = (msg) => {
    if (!msg || msg.type !== "lg:res") return;
    const entry = pending.get(msg.id);
    if (!entry) return;
    pending.delete(msg.id);
    if (msg.ok) entry.resolve(msg.result);
    else entry.reject(new Error(msg.error || "[lattice] the Python engine failed"));
  };
  model.on("msg:custom", onMsg);
  const ask = (method, payload, signal) => new Promise((resolve, reject) => {
    if (signal && signal.aborted) { reject(signal.reason || new Error("aborted")); return; }
    const id = ++seq;
    pending.set(id, { resolve, reject });
    if (signal) {
      signal.addEventListener("abort", () => {
        if (pending.delete(id)) reject(signal.reason || new Error("aborted"));
      }, { once: true });
    }
    model.send({ type: "lg:req", id, method, payload });
  });
  ask.dispose = () => { model.off("msg:custom", onMsg); pending.clear(); };
  ask.pending = () => pending.size;
  return ask;
}

/** A query as plain JSON (no signal, no functions). */
const plain = (q) => JSON.parse(JSON.stringify(q, (k, v) => (k === "signal" ? undefined : v)));

/** The pushdown adapter whose engine is Python (the grid's adapter contract). */
function pythonAdapter(ask, caps) {
  return {
    name: "python",
    capabilities: {
      filter: "tree",
      operators: caps.operators || [],
      sort: "multi",
      quick: true,
      range: true,
      total: true,
      group: false,
      pivot: false,
      aggregates: caps.aggregates || [],
      mutate: caps.editable ? { update: true, returning: "none" } : false,
    },
    async execute(query, request) {
      const r = await ask("execute", { query: plain(query) }, request && request.signal);
      return { rows: rowsFromColumnar(r.columnar), total: r.total };
    },
    executeGroupedAggregates(query, groupBy, aggregates, opts) {
      return ask("grouped", { query: plain(query), groupBy: plain(groupBy), aggregates: plain(aggregates) },
        opts && opts.signal);
    },
    executeAggregates(query, aggregates) {
      return ask("aggregates", { query: plain(query), aggregates: plain(aggregates) });
    },
    async mutate(op, request) {
      const r = await ask("mutate", { op: plain(op) }, request && request.signal);
      return r || { ok: true };
    },
  };
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

  // --- large frames: a windowed source answered by Python (card 1618) --------
  const ask = createCommClient(model);
  const windowed = () => model.get("_mode") === "windowed";
  const pageSize = () => model.get("_page_size") || 100;
  const pushdown = () => mod.createPushdownSource({
    adapter: pythonAdapter(ask, model.get("_engine_caps") || {}),
    pageSize: pageSize(),
    edit: !!(model.get("_engine_caps") || {}).editable,
  });
  // Header histograms over a windowed source need counts the browser cannot
  // compute: the grid asks its `facets.provider`, which asks Python.
  const facetProvider = (req) => ask("facet", {
    colId: req.colId, filters: req.filters ? plain(req.filters) : null, quick: req.quick || "",
    bounds: req.bounds || null, buckets: req.buckets, strategy: req.strategy, granularity: req.granularity,
  });
  const withProvider = (opts) => {
    if (!windowed() || !opts.facets) return opts;
    const f = opts.facets === true ? { enabled: true } : opts.facets;
    return { ...opts, facets: { ...f, provider: facetProvider } };
  };
  const dataConfig = () => (windowed()
    ? { source: pushdown() }
    : { rows: reconstructRows(model) });

  const grid = createGrid(host, {
    selection: "multiple",
    edit: true,
    ...withProvider(appliedOptions),
    // Python owns the data, the row key and the column set.
    columns: model.get("_columns"),
    ...dataConfig(),
    rowKey: model.get("_row_key") || "__row_id__",
    ...(initialState && Object.keys(initialState).length ? { state: initialState } : {}),
    ...(licence ? { licence } : {}),
  });
  host.__latticeGrid = grid; // handle for tests and notebook-side debugging
  host.__latticeAsk = ask;

  // Profile statistics over a windowed source: the grid's own profile reads the
  // rows it holds (a window), so the Statistics panel's profile is answered by
  // Python over the whole frame instead. Shadowed on this grid instance only,
  // never on the prototype; every other statistic is the grid's own.
  const profiles = new Map();
  let profilesFor = "";
  const profileScope = () => JSON.stringify([grid.state.get().filters || null, grid.state.get().quick || ""]);
  const repaintStats = () => {
    const dock = grid.toolPanel;
    const open = host.querySelector('[data-panel="statistics"], .lat-statistics');
    if (dock && typeof dock.open === "function" && open) dock.open("statistics");
  };
  const realStatistics = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(grid), "statistics")
    || Object.getOwnPropertyDescriptor(Object.getPrototypeOf(Object.getPrototypeOf(grid)), "statistics");
  if (realStatistics && realStatistics.get) {
    Object.defineProperty(grid, "statistics", {
      configurable: true,
      get() {
        const real = realStatistics.get.call(grid);
        if (!windowed()) return real;
        return {
          ...real,
          profile(colId) {
            const scope = profileScope();
            if (scope !== profilesFor) { profiles.clear(); profilesFor = scope; }
            const hit = profiles.get(colId);
            if (hit && hit !== "pending") return hit;
            if (!hit) {
              profiles.set(colId, "pending");
              const st = grid.state.get();
              ask("profile", { colId, filters: st.filters ? plain(st.filters) : null, quick: st.quick || "" })
                .then((p) => { if (profilesFor === scope) { profiles.set(colId, p); repaintStats(); } })
                .catch(() => { if (profiles.get(colId) === "pending") profiles.delete(colId); });
            }
            return null;
          },
        };
      },
    });
  }

  // Surface the resolved licence state for debugging / the smoke test.
  try {
    if (grid.licence && typeof grid.licence.state === "function") {
      model.set("_licence_state", String(grid.licence.state()));
      model.save_changes();
    }
  } catch (e) { /* non-fatal */ }

  // --- grid edit -> DataFrame -------------------------------------------------
  const onCellChanged = (e) => {
    // Over the windowed source an edit reaches Python through the adapter's
    // `mutate` (the pushdown write-back contract), not through this trait.
    if (windowed()) return;
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
  let shownMode = model.get("_mode") || "client";
  const onData = () => {
    const mode = model.get("_mode") || "client";
    profiles.clear();
    if (mode === "windowed") {
      // A new windowed source: every cached window, count and histogram goes.
      grid.set("source", pushdown());
    } else if (shownMode === "windowed") {
      grid.set("source", { mode: "memory", rows: reconstructRows(model) });
    } else if (typeof grid.rows.load === "function") {
      // load() keeps the column layout; set_data assumes the same schema.
      grid.rows.load(reconstructRows(model));
    }
    shownMode = mode;
  };
  model.on("change:_data_version", onData);

  // --- Python append_rows / delete_rows -> incremental apply -----------------
  const onRowOp = () => {
    const op = model.get("_row_op");
    if (!op || !op.op) return;
    if (windowed()) return; // Python re-sources the windowed grid itself (_data_version)
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
    // Windowed: the view is the whole frame under `state`; Python computes it
    // lazily from the state it already has. No keys cross the comm.
    if (windowed()) return;
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
    if ("facets" in values && values.facets) values.facets = withProvider({ facets: values.facets }).facets;
    if (typeof grid.setAll === "function") grid.setAll(values);
    else for (const k of Object.keys(values)) grid.set(k, values[k]);
    setTimeout(reportWarnings, 0);
  };
  model.on("change:_grid_options", onOptions);

  return () => {
    ask.dispose();
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
