/**
 * Browser bundle entry.
 *
 * Dash loads each component library's JS and expects the component classes to be
 * exposed on `window[<namespace>]`. `dash-generate-components` writes Python
 * classes whose `_js_dist` points at the file esbuild produces from this entry.
 */
import {LatticeGrid} from './index';

/* eslint-disable no-undef */
window.lattice_grid_dash = Object.assign({}, window.lattice_grid_dash, {
    LatticeGrid,
});
