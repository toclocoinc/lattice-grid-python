// Parity harness (BACKLOG-0005044): the vendored data-router bundle, driven the way the
// widget's front end drives it (static/widget.js: withSelectFns, routerRows, startRouter and
// reportShaped), with no grid and no browser. Reads {bundle, payload} as JSON on stdin and
// writes the columnar rows the widget would report to Python as `_shaped`.
const fs = require("fs");

function withSelectFns(options) {
  const fix = (j) => {
    if (!j || typeof j !== "object") return j;
    const out = { ...j };
    if (out.many && out.as && out.select !== undefined && typeof out.select !== "function") {
      const sel = out.select;
      out.select = typeof sel === "string" ? (row) => row[sel]
        : (row) => Object.fromEntries((Array.isArray(sel) ? sel : Object.keys(sel)).map((f) => [Array.isArray(sel) ? f : sel[f], row[f]]));
    }
    return out;
  };
  const joins = (js) => (Array.isArray(js) ? js.map(fix) : fix(js));
  const unnests = (us) => (Array.isArray(us) ? us.map(unnest) : unnest(us));
  const unnest = (u) => ({ ...u, ...(u.join ? { join: joins(u.join) } : {}), ...(u.unnest ? { unnest: unnests(u.unnest) } : {}) });
  return { ...options, ...(options.join ? { join: joins(options.join) } : {}), ...(options.unnest ? { unnest: unnests(options.unnest) } : {}) };
}

function routerRows(columnar, sourceId) {
  const keys = (columnar && columnar["__row_id__"]) || [];
  const fields = Object.keys(columnar || {}).filter((f) => f !== "__row_id__");
  const rows = new Array(keys.length);
  for (let i = 0; i < keys.length; i++) {
    const rec = { __row_id__: keys[i], __source: sourceId };
    for (let c = 0; c < fields.length; c++) rec[fields[c]] = columnar[fields[c]][i];
    rows[i] = rec;
  }
  return rows;
}

let input = "";
process.stdin.on("data", (d) => { input += d; });
process.stdin.on("end", () => {
  const { bundle, payload } = JSON.parse(input);
  const lib = require(bundle);
  const spec = payload;
  const router = lib.createDataRouter({ key: "__source", rowKey: "__row_id__" });
  const mirror = new Map();
  router.subscribe(spec.route, (change) => {
    for (const r of change.add || []) mirror.set(String(r.__row_id__), r);
    for (const r of change.update || []) mirror.set(String(r.__row_id__), r);
    for (const k of change.remove || []) mirror.delete(String(k));
  });
  const handles = {};
  const ids = Object.keys(spec.sources || {});
  for (const id of ids) handles[id] = router.addSource(id, { key: true, ...withSelectFns(spec.sources[id].options || {}) });
  for (const id of spec.loadOrder || ids) handles[id].load(routerRows(spec.sources[id].columnar, id));
  const rows = [...mirror.values()];
  const fields = [];
  const seen = new Set();
  for (const r of rows) for (const f of Object.keys(r)) if (f !== "__row_id__" && f !== "__source" && !seen.has(f)) { seen.add(f); fields.push(f); }
  const columnar = { __row_id__: rows.map((r) => String(r.__row_id__)) };
  for (const f of fields) columnar[f] = rows.map((r) => (r[f] === undefined ? null : r[f]));
  process.stdout.write(JSON.stringify(columnar));
});
