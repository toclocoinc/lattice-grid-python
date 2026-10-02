/*
 * lattice-grid-jupyter -- the chart widget's front end (card 1619).
 *
 * A chart is a viewer of a dataset. Two ways to get one:
 *   mode "grid"  -> bind to the LIVE grid another widget rendered (found by uid
 *                   in window.__latticeGrids, which widget.js fills). The chart
 *                   reads the grid's rows and redraws on the grid's own events,
 *                   so a filter, sort or selection reaches it with no round trip
 *                   to Python.
 *   mode "frame" -> a DataFrame sent as columns; a hidden grid holds it and the
 *                   chart binds to that, through the same binding.
 *
 * A WINDOWED grid holds one window of its rows (the rest stay in Python). The
 * chart then draws from the grouped aggregates the grid's engine already
 * answers (the same `grouped` request the grid's own subtotals use) and never
 * reads rows; types that need the rows themselves are refused by Python
 * (model._refused) and show that reason instead of a picture.
 *
 * Code loads only when used: the base charts module plus the one module the
 * type needs. model._chart_js carries their UMD text (offline); otherwise they
 * come from jsDelivr as classic scripts (the ESM builds import a bare specifier
 * a notebook page cannot resolve).
 */

const CDN_BASE = (v) => `https://cdn.jsdelivr.net/npm/@toclocoinc/lattice-grid@${v}`;

function injectScriptText(text) {
  const s = document.createElement("script");
  s.textContent = text;
  document.head.appendChild(s);
}

function loadScriptUrl(url) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = url;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`[lattice] could not load ${url}`));
    document.head.appendChild(s);
  });
}

/** The grid library, for the hidden grid of a DataFrame-backed chart. */
async function loadGrid(model) {
  if (model.get("_grid_source") === "vendor") {
    if (!document.querySelector("style[data-lattice-css]")) {
      const style = document.createElement("style");
      style.setAttribute("data-lattice-css", "1");
      style.textContent = model.get("_grid_css") || "";
      document.head.appendChild(style);
    }
    if (!(window.LatticeGrid && typeof window.LatticeGrid.createGrid === "function")) {
      injectScriptText(model.get("_grid_js") || "");
    }
    return window.LatticeGrid;
  }
  const base = CDN_BASE(model.get("_grid_version"));
  if (!document.querySelector("link[data-lattice-css]")) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = base + "/lattice-grid.min.css";
    link.setAttribute("data-lattice-css", "1");
    document.head.appendChild(link);
  }
  if (!window.__latticeGridMod) window.__latticeGridMod = import(base + "/lattice-grid.esm.min.js");
  return window.__latticeGridMod;
}

/**
 * Make sure the base chart module and the named type modules are loaded, once
 * per page. Resolves false when a module's text has not arrived yet (Python
 * sends the type and its module in separate messages); the next change retries.
 */
async function loadModules(model, names) {
  const loaded = (window.__latticeChartModules = window.__latticeChartModules || new Map());
  const vendor = model.get("_grid_source") === "vendor";
  const texts = model.get("_chart_js") || {};
  for (const name of ["charts", ...names]) {
    if (loaded.has(name)) { await loaded.get(name); continue; }
    if (vendor) {
      if (!texts[name]) return false;
      injectScriptText(texts[name]);
      loaded.set(name, Promise.resolve());
    } else {
      const p = loadScriptUrl(`${CDN_BASE(model.get("_grid_version"))}/modules/${name}.min.js`);
      loaded.set(name, p);
      try { await p; } catch (e) { loaded.delete(name); throw e; }
    }
  }
  return true;
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

const registry = () => (window.__latticeGrids = window.__latticeGrids || { grids: new Map(), listeners: new Map() });

async function render({ model, el }) {
  const wrap = document.createElement("div");
  wrap.style.width = "100%";
  const note = document.createElement("div");
  note.setAttribute("role", "status");
  note.style.cssText = "font:13px sans-serif;color:#555;padding:6px 2px;";
  const plot = document.createElement("div");
  plot.style.cssText = `width:100%;height:${model.get("height") || 320}px;`;
  wrap.append(note, plot);
  el.appendChild(wrap);

  let chart = null;
  let hidden = null;            // the hidden grid of a DataFrame-backed chart
  let hiddenHost = null;
  let boundGrid = null;
  let offGrid = [];             // unsubscribers for the bound grid
  let draws = 0;
  let generation = 0;
  let destroyed = false;
  let timer = null;
  let aggKey = "";
  let aggBusy = false;
  const frame = () => model.get("_mode") === "frame";

  const fail = (e) => {
    const text = String((e && e.message) || e);
    note.textContent = text;
    model.set("_chart_error", text);
    model.save_changes();
  };

  const reportDraws = () => {
    plot.__draws = draws;
    model.set("_draws", draws);
    model.save_changes();
  };
  let reportTimer = null;
  const onDraw = () => {
    draws += 1;
    plot.__draws = draws;
    if (reportTimer === null) reportTimer = setTimeout(() => { reportTimer = null; reportDraws(); }, 120);
  };

  const teardown = () => {
    for (const off of offGrid) { try { off(); } catch (e) { /* ignore */ } }
    offGrid = [];
    if (chart) { try { chart.destroy(); } catch (e) { /* ignore */ } chart = null; }
    plot.__latticeChart = null;
    aggKey = "";
  };

  // The grid this chart reads: the live grid another widget rendered, or a hidden
  // one holding a DataFrame.
  const subscribers = registry().listeners;
  const uid = model.get("_grid_uid");
  let entry = null;
  const listener = (e) => { entry = e; schedule(); };
  if (uid) {
    if (!subscribers.has(uid)) subscribers.set(uid, new Set());
    subscribers.get(uid).add(listener);
    entry = registry().grids.get(uid) || null;
  }

  async function hiddenGrid() {
    const mod = await loadGrid(model);
    const rows = rowsFromColumnar(model.get("_columnar"));
    if (!hidden) {
      hiddenHost = document.createElement("div");
      hiddenHost.style.cssText = "position:absolute;left:-10000px;top:0;width:400px;height:200px;overflow:hidden;";
      wrap.appendChild(hiddenHost);
      hidden = mod.createGrid(hiddenHost, {
        columns: model.get("_columns"), rows, rowKey: model.get("_row_key") || "__row_id__",
        ...(model.get("licence") ? { licence: model.get("licence") } : {}),
      });
    }
    return hidden;
  }

  /** One value per category (per series) from the engine, under the grid's own filters. */
  async function aggregateRows(ask, grid, spec) {
    const x = spec.x, y = spec.y, series = spec.series;
    const fn = (spec.measures && spec.measures[0] && spec.measures[0].fn) || "sum";
    const st = grid.state.get();
    const groupBy = series ? [x, series] : [x];
    const groups = await ask("grouped", {
      query: { filters: st.filters || null, quick: st.quick || "" },
      groupBy,
      aggregates: [{ id: "v", col: y, fn }],
    });
    return groups.filter((g) => g.level === groupBy.length).map((g, i) => {
      const r = { __row_id__: String(i) };
      groupBy.forEach((k, j) => { r[k] = g.keys[j]; });
      r[y] = g.values.v;
      return r;
    });
  }

  async function apply() {
    const mine = ++generation;
    const type = model.get("type");
    const refused = model.get("_refused") || "";
    teardown();
    note.textContent = "";
    if (refused) { note.textContent = refused; return; }
    // The grid library goes first: its UMD build publishes window.LatticeGrid and
    // would replace the chart exports a module merged there before it.
    let grid;
    let entryNow = null;
    if (frame()) grid = await hiddenGrid();
    const ready = await loadModules(model, model.get("_modules") || []);
    if (!ready || mine !== generation || destroyed) return;
    if (!frame()) {
      entryNow = entry || registry().grids.get(uid) || null;
      if (!entryNow) { note.textContent = "Waiting for the bound grid to be displayed..."; return; }
      grid = entryNow.grid;
    }
    if (mine !== generation || destroyed) return;
    boundGrid = grid;
    const spec = { ...(model.get("_spec") || {}) };
    const windowed = !frame() && entryNow && entryNow.windowed();
    const createChart = window.LatticeGrid && window.LatticeGrid.createChart;
    if (typeof createChart !== "function") throw new Error("[lattice] the chart module did not load");

    if (windowed && !["bar", "horizontalBar", "line", "step", "area", "pie", "donut", "funnel", "waterfall"].includes(type)) {
      note.textContent = `the grid is windowed, so a ${type} chart is not drawn from a window of its rows`;
      return;
    }
    let chartGrid = grid;
    let aggGrid = null;
    if (windowed) {
      // The dataset this chart views is the grid's aggregate: one row per category (per
      // series), computed in Python over the WHOLE frame under the grid's filters. It
      // lives in a small hidden grid; the chart binds to that through the same binding.
      const rows = await aggregateRows(entryNow.ask, grid, spec);
      if (mine !== generation || destroyed) return;
      const cols = [spec.x, ...(spec.series ? [spec.series] : [])].map((f) => {
        const def = grid.columns.get(f);
        return { field: f, title: (def && def.title) || f, type: (def && def.type) || "text" };
      }).concat([{ field: spec.y, title: spec.y, type: "number" }]);
      const host = document.createElement("div");
      host.style.cssText = "position:absolute;left:-10000px;top:0;width:400px;height:200px;overflow:hidden;";
      wrap.appendChild(host);
      aggGrid = entryNow.lib.createGrid(host, { columns: cols, rows, rowKey: "__row_id__" });
      chartGrid = aggGrid;
      offGrid.push(() => { try { aggGrid.destroy(); } catch (e) { /* ignore */ } host.remove(); });
    }
    chart = createChart({ ...spec, type, grid: chartGrid, container: plot });
    plot.__latticeChart = chart;
    chart.on("draw", onDraw);
    if (chart.data()) onDraw();  // createChart may have drawn before the listener existed

    if (windowed) {
      // Re-aggregate when the question changes (filter, quick search, row count), not on every scroll.
      aggKey = JSON.stringify([grid.state.get().filters || null, grid.state.get().quick || "", grid.rows.matchCount && grid.rows.matchCount()]);
      const recheck = async () => {
        if (aggBusy || !chart) return;
        const st = grid.state.get();
        const key = JSON.stringify([st.filters || null, st.quick || "", grid.rows.matchCount && grid.rows.matchCount()]);
        if (key === aggKey) return;
        aggKey = key;
        aggBusy = true;
        try {
          const rows = await aggregateRows(entryNow.ask, grid, spec);
          if (chart && mine === generation) aggGrid.rows.load(rows);
        } catch (e) { fail(e); } finally { aggBusy = false; }
        if (JSON.stringify([grid.state.get().filters || null, grid.state.get().quick || "", grid.rows.matchCount && grid.rows.matchCount()]) !== aggKey) recheck();
      };
      let t = null;
      const debounced = () => { if (t !== null) clearTimeout(t); t = setTimeout(() => { t = null; recheck(); }, 120); };
      grid.on("state:changed", debounced);
      grid.on("model:changed", debounced);
      offGrid.push(() => { if (t !== null) clearTimeout(t); grid.off && grid.off("state:changed", debounced); grid.off && grid.off("model:changed", debounced); });
    }
  }

  function schedule() {
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(() => { timer = null; apply().catch(fail); }, 0);
  }

  const watch = ["change:type", "change:_spec", "change:_modules", "change:_chart_js", "change:_refused", "change:_windowed"];
  for (const ev of watch) model.on(ev, schedule);
  const onData = async () => {
    if (!frame() || !hidden) return;
    hidden.rows.load(rowsFromColumnar(model.get("_columnar")));
    if (model.get("_refused")) schedule();
  };
  model.on("change:_data_version", onData);

  try { await apply(); } catch (e) { fail(e); }

  return () => {
    destroyed = true;
    generation += 1;
    if (timer !== null) clearTimeout(timer);
    if (reportTimer !== null) clearTimeout(reportTimer);
    for (const ev of watch) model.off(ev, schedule);
    model.off("change:_data_version", onData);
    if (uid && subscribers.get(uid)) subscribers.get(uid).delete(listener);
    teardown();
    if (hidden && hidden.destroy) hidden.destroy();
  };
}

export default { render };
