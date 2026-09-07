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

  // CDN mode: dynamic ESM import, deduped on the window.
  const version = model.get("_grid_version") || "1.40.0";
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
  const grid = createGrid(host, {
    columns: model.get("_columns"),
    rows: reconstructRows(model),
    rowKey: model.get("_row_key") || "__row_id__",
    edit: true,
    ...(licence ? { licence } : {}),
  });

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

  return () => {
    model.off("change:_data_version", onData);
    model.off("change:_row_op", onRowOp);
    grid.off && grid.off("cell:changed", onCellChanged);
    grid.destroy && grid.destroy();
  };
}

export default { render };
