/**
 * Build the browser bundle for the Dash component.
 *
 * Output: lattice_grid_dash/lattice_grid_dash.min.js  (an IIFE that registers
 * window.lattice_grid_dash.LatticeGrid).
 *
 * DELIVERY = "vendor" (implemented): the grid core + its React adapter are
 * bundled in, so the component works fully offline with no CDN at render time.
 * See README for the "cdn" delivery variant (mark the grid packages external and
 * load them from jsDelivr at runtime) -- documented, not built here.
 *
 * React/ReactDOM are NOT bundled: the component reads window.React (served by the
 * dash-renderer), guaranteeing a single React instance.
 */
import {build} from 'esbuild';
import {mkdirSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {dirname, resolve} from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
mkdirSync('lattice_grid_dash', {recursive: true});

await build({
    entryPoints: ['src/lib/bundle.js'],
    bundle: true,
    format: 'iife',
    target: ['es2019'],
    minify: true,
    sourcemap: false,
    outfile: 'lattice_grid_dash/lattice_grid_dash.min.js',
    // `react` -> Dash's global React (see src/lib/react-shim.js); react-dom is
    // unused by the grid/adapter, so exclude it entirely.
    alias: {react: resolve(__dirname, 'src/lib/react-shim.js')},
    external: ['react-dom'],
    logLevel: 'info',
});

console.log('built lattice_grid_dash/lattice_grid_dash.min.js');
