"""Minimal Dash app demonstrating the LatticeGrid component round-trip.

Run:  python examples/app.py   then open http://127.0.0.1:8050

Edit a cell in the grid; the panel below updates from a Python @callback that
observes the component's `cellChanged` prop, and the DataFrame is updated
server-side (dtype-preserved) via `apply_cell_edit`.
"""

import pandas as pd
from dash import Dash, Input, Output, callback, html

import lattice_grid_dash
from lattice_grid_dash import apply_cell_edit, dataframe_to_data

df = pd.DataFrame(
    {
        "name": ["Ada", "Grace", "Linus"],
        "score": [91, 88, 77],
        "active": [True, False, True],
    }
)

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H3("Lattice Grid in Dash"),
        lattice_grid_dash.LatticeGrid(
            id="grid",
            data=dataframe_to_data(df),
            options={"edit": True},
        ),
        html.Pre(id="out", children="Edit a cell..."),
    ],
    style={"maxWidth": "720px", "margin": "2rem auto", "fontFamily": "system-ui"},
)


@callback(Output("out", "children"), Input("grid", "cellChanged"))
def on_edit(edit):
    if not edit:
        return "Edit a cell..."
    apply_cell_edit(df, edit)  # server-side DataFrame stays in sync + typed
    return (
        f"cellChanged -> row {edit['key']}, column '{edit['colId']}' = "
        f"{edit['value']!r} (was {edit.get('old')!r})\n"
        f"df row now: {df.iloc[int(edit['key'])].to_dict()}"
    )


if __name__ == "__main__":
    app.run(debug=True)
