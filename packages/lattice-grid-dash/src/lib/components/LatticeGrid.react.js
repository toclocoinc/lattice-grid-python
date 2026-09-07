/**
 * LatticeGrid -- a Dash component wrapping Lattice Grid.
 *
 * Rendering path (lowest-risk, per the C2 spike): we do NOT re-implement a React
 * binding. We consume the grid's *shipped* React adapter
 * (`@toclocoinc/lattice-grid/modules/react`), which exposes
 * `createLatticeGrid({ React, createGrid })` -> a forwardRef React component. We
 * feed it Dash's own React (the global the dash-renderer serves) so there is a
 * single React instance, and `createGrid` from the published grid package.
 *
 * Dash <-> grid bridge:
 *   - `data` (columnar, produced by lattice-grid-pandas) is reconstructed into
 *     row objects here and handed to the adapter.
 *   - a manual cell edit -> the adapter's `onCellChanged` -> `setProps` updates
 *     the `cellChanged` prop, which a Python `@callback` observes.
 *   - selection -> `onSelectionChanged` -> `setProps({selectedKeys})`.
 *   - the grid's resolved licence state -> `setProps({licenceState})`.
 *
 * We deliberately import nothing from 'react': Dash owns React and serves it as a
 * global, so we read it off the window. That keeps a single React instance (no
 * duplicate-React hook errors) and needs no react/react-dom in this bundle.
 */

import React from 'react';
import PropTypes from 'prop-types';
import createLatticeGrid from '@toclocoinc/lattice-grid/modules/react';
import {createGrid} from '@toclocoinc/lattice-grid';

// NOTE: at bundle time `react` is aliased to Dash's global React (see build.mjs),
// so this is Dash's own React instance -- a single React across the page, no
// duplicate-React hook errors, and no react/react-dom bundled here.

// Build the shipped adapter component once, bound to Dash's React + createGrid.
const LatticeGridAdapter = createLatticeGrid({React, createGrid});

const ROW_KEY = '__row_id__';

/**
 * Rebuild row objects from the columnar payload lattice-grid-pandas emits:
 *   { "__row_id__": [k0, k1, ...], field: [v0, v1, ...], ... }
 * (identical reconstruction to the Jupyter widget's front end.)
 */
function reconstructRows(columnar) {
    if (!columnar) return [];
    const keys = columnar[ROW_KEY] || [];
    const fields = Object.keys(columnar).filter((f) => f !== ROW_KEY);
    const n = keys.length;
    const rows = new Array(n);
    for (let i = 0; i < n; i++) {
        const rec = {};
        rec[ROW_KEY] = keys[i];
        for (let c = 0; c < fields.length; c++) {
            rec[fields[c]] = columnar[fields[c]][i];
        }
        rows[i] = rec;
    }
    return rows;
}

/**
 * LatticeGrid Dash component.
 */
function LatticeGrid(props) {
    const {
        id,
        data,
        columns,
        options,
        licence,
        selectedKeys,
        style,
        className,
        setProps,
    } = props;

    const gridRef = React.useRef(null);
    const rows = React.useMemo(
        () => reconstructRows(data && data.columnar),
        [data]
    );
    const cols = columns || (data && data.columns) || [];
    const rowKey = (data && data.rowKey) || ROW_KEY;

    // Surface the grid's resolved licence state back to Python once it exists.
    React.useEffect(() => {
        const t = setTimeout(() => {
            try {
                const g = gridRef.current && gridRef.current.grid;
                if (g && g.licence && typeof g.licence.state === 'function' && setProps) {
                    setProps({licenceState: String(g.licence.state())});
                }
            } catch (e) {
                /* non-fatal */
            }
        }, 0);
        return () => clearTimeout(t);
        // rows in the dep list so it re-reads after a (re)render/reload.
    }, [rows, setProps]);

    const adapterProps = {
        ref: gridRef,
        id,
        columns: cols,
        rows,
        rowKey,
        edit: true,
        ...(options || {}),
        ...(licence ? {licence} : {}),
        ...(Array.isArray(selectedKeys) ? {selectedKeys} : {}),
        style: {height: '360px', width: '100%', ...(style || {})},
        className,

        // grid -> Dash
        onCellChanged: (e) => {
            if (!setProps) return;
            setProps({
                cellChanged: {
                    key: String(e.key),
                    colId: e.colId,
                    value: e.value,
                    old: e.oldValue,
                    ts: Date.now(),
                },
            });
        },
        onSelectionChanged: (e) => {
            if (!setProps) return;
            const keys = (e && e.keys) || [];
            setProps({selectedKeys: keys.map(String)});
        },
        onLicenceChanged: (e) => {
            if (!setProps) return;
            const state = (e && (e.state || e.status)) || '';
            if (state) setProps({licenceState: String(state)});
        },
    };

    return React.createElement(LatticeGridAdapter, adapterProps);
}

LatticeGrid.defaultProps = {
    data: {columns: [], rowKey: ROW_KEY, columnar: {}},
    options: {},
    licence: '',
};

LatticeGrid.propTypes = {
    /** Component id, used to target the component in Dash callbacks. */
    id: PropTypes.string,

    /**
     * Columnar, DataFrame-derived data produced by `lattice-grid-pandas`:
     * `{columns: [...], rowKey: "__row_id__", columnar: {field: [values...]}}`.
     * Rows are reconstructed in the browser (virtualized rendering).
     */
    data: PropTypes.object,

    /** Optional explicit column definitions; overrides `data.columns`. */
    columns: PropTypes.array,

    /** Grid options passed through to `createGrid` (e.g. edit, rowHeight). */
    options: PropTypes.object,

    /** Lattice Grid licence key. Empty on localhost runs free/unwatermarked. */
    licence: PropTypes.string,

    /**
     * OUTPUT. The last cell edit made in the grid:
     * `{key, colId, value, old, ts}`. Updated on every manual edit so a Python
     * `@callback` with `Input(id, "cellChanged")` fires. Read-only from Python.
     */
    cellChanged: PropTypes.object,

    /** Selected row keys. Set by the grid on selection; may also be set from Python. */
    selectedKeys: PropTypes.arrayOf(PropTypes.string),

    /** OUTPUT. The grid's resolved licence state (e.g. "localhost", "valid"). */
    licenceState: PropTypes.string,

    /** Inline style for the grid host element. */
    style: PropTypes.object,

    /** CSS class for the grid host element. */
    className: PropTypes.string,

    /** Dash-assigned callback for updating props. */
    setProps: PropTypes.func,
};

export default LatticeGrid;
