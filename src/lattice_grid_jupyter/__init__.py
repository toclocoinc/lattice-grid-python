"""lattice-grid-jupyter -- an editable Lattice Grid over a pandas DataFrame in Jupyter."""

from ._options import LatticeGridWarning
from .widget import LatticeGridWidget, GRID_VERSION

__all__ = ["LatticeGridWidget", "LatticeGridWarning", "GRID_VERSION", "__version__"]
__version__ = "1.85.0.0"
