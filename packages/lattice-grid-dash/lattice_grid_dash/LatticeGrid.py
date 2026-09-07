# AUTO GENERATED FILE - DO NOT EDIT

import typing  # noqa: F401
from typing_extensions import TypedDict, NotRequired, Literal # noqa: F401
from dash.development.base_component import Component, _explicitize_args
try:
    from dash.types import NumberType  # noqa: F401
except ImportError:
    # Backwards compatibility for dash<=4.1.0
    if typing.TYPE_CHECKING:
        raise
    NumberType = typing.Union[  # noqa: F401
        typing.SupportsFloat, typing.SupportsInt, typing.SupportsComplex
    ]

ComponentSingleType = typing.Union[str, int, float, Component, None]
ComponentType = typing.Union[
    ComponentSingleType,
    typing.Sequence[ComponentSingleType],
]


class LatticeGrid(Component):
    """A LatticeGrid component.
LatticeGrid Dash component.

Keyword arguments:

- id (string; optional):
    Component id, used to target the component in Dash callbacks.

- cellChanged (dict; optional):
    OUTPUT. The last cell edit made in the grid: `{key, colId, value,
    old, ts}`. Updated on every manual edit so a Python `@callback`
    with `Input(id, \"cellChanged\")` fires. Read-only from Python.

- className (string; optional):
    CSS class for the grid host element.

- columns (list; optional):
    Optional explicit column definitions; overrides `data.columns`.

- data (dict; default {columns: [], rowKey: ROW_KEY, columnar: {}}):
    Columnar, DataFrame-derived data produced by
    `lattice-grid-pandas`: `{columns: [...], rowKey: \"__row_id__\",
    columnar: {field: [values...]}}`. Rows are reconstructed in the
    browser (virtualized rendering).

- licence (string; default ''):
    Lattice Grid licence key. Empty on localhost runs
    free/unwatermarked.

- licenceState (string; optional):
    OUTPUT. The grid's resolved licence state (e.g. \"localhost\",
    \"valid\").

- options (dict; optional):
    Grid options passed through to `createGrid` (e.g. edit,
    rowHeight).

- selectedKeys (list of strings; optional):
    Selected row keys. Set by the grid on selection; may also be set
    from Python."""
    _children_props: typing.List[str] = []
    _base_nodes = ['children']
    _namespace = 'lattice_grid_dash'
    _type = 'LatticeGrid'


    def __init__(
        self,
        id: typing.Optional[typing.Union[str, dict]] = None,
        data: typing.Optional[dict] = None,
        columns: typing.Optional[typing.Sequence] = None,
        options: typing.Optional[dict] = None,
        licence: typing.Optional[str] = None,
        cellChanged: typing.Optional[dict] = None,
        selectedKeys: typing.Optional[typing.Sequence[str]] = None,
        licenceState: typing.Optional[str] = None,
        style: typing.Optional[typing.Any] = None,
        className: typing.Optional[str] = None,
        **kwargs
    ):
        self._prop_names = ['id', 'cellChanged', 'className', 'columns', 'data', 'licence', 'licenceState', 'options', 'selectedKeys', 'style']
        self._valid_wildcard_attributes =            []
        self.available_properties = ['id', 'cellChanged', 'className', 'columns', 'data', 'licence', 'licenceState', 'options', 'selectedKeys', 'style']
        self.available_wildcard_properties =            []
        _explicit_args = kwargs.pop('_explicit_args')
        _locals = locals()
        _locals.update(kwargs)  # For wildcard attrs and excess named props
        args = {k: _locals[k] for k in _explicit_args}

        super(LatticeGrid, self).__init__(**args)

setattr(LatticeGrid, "__init__", _explicitize_args(LatticeGrid.__init__))
