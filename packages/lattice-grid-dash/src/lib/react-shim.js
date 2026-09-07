/**
 * Bundle-time shim: resolve `react` to the global React the dash-renderer serves.
 *
 * build.mjs aliases the bare `react` specifier to this file, so `import React
 * from 'react'` in the component resolves to Dash's own React instance at
 * runtime. This keeps a single React on the page (no duplicate-React hook
 * errors) and keeps react/react-dom out of this component bundle, exactly as the
 * standard Dash `externals: {react: 'React'}` webpack config does.
 */
/* eslint-disable no-undef */
const R = (typeof window !== 'undefined' && window.React) || null;
if (!R) {
    throw new Error('[lattice-grid-dash] window.React not found; Dash serves it.');
}
export default R;
