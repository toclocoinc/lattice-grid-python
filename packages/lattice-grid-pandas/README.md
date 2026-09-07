# lattice-grid-pandas

Framework-agnostic glue between a pandas `DataFrame` and [Lattice Grid](https://latticegrid.dev).

This package is the **single source of truth** for the parts every Python host
wrapper needs and must not re-implement:

- the pandas-dtype -> grid column `type` mapping (`lattice_type`),
- column-major (columnar) serialization of a frame the browser reconstructs
  (`build_columns`, `build_columnar`, `column_values`),
- value/dtype coercion for the edit round-trip (`cast_to_dtype`), and
- grid delivery + licence plumbing (`GRID_VERSION`, `cdn_urls`, `load_vendored`,
  `resolve_source`).

It has no UI and no framework dependency -- only pandas -- so it can be unit
tested without a browser or a notebook kernel.

Consumers:

- `lattice-grid-jupyter` (phase C1) -- anywidget notebook widget.
- `lattice-grid-dash` (phase C2) -- Dash component.
