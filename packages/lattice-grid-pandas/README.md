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

## Datetime contract: UTC end to end

**Naive `datetime64` columns are UTC.** A column with no tz info is treated as
already being UTC, not the machine's local zone: outbound, its values
serialize as ISO 8601 strings with an explicit `Z` (e.g.
`2021-06-01T12:00:00Z`), never a bare `2021-06-01T12:00:00` -- the grid's
`timestamp` type parses a bare string as a *wall-clock* time in the browser's
own zone, which would silently shift every value by the browser's UTC offset.
A tz-aware column is converted to UTC before serializing, so every host
renders the identical instant regardless of the column's original zone.
Inbound, an edit is accepted as a `Number` (the grid's epoch-**millisecond**
instant), an ISO string with or without an offset (an offset-less string is
UTC under this same contract), or a `pd.Timestamp`; the result is converted to
UTC and then either stripped of tz (a naive column stays naive) or converted
to the column's own tz (a tz-aware column keeps reading in its own zone).
`NaT` survives the round trip as JSON `null` both ways.
