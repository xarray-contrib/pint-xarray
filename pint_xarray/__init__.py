from importlib.metadata import version

import pint

from pint_xarray import accessors, formatting, testing  # noqa: F401
from pint_xarray.accessors import default_registry as unit_registry
from pint_xarray.accessors import setup_registry
from pint_xarray.index import PintIndex

try:
    __version__ = version("pint-xarray")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"


pint.Quantity._repr_inline_ = formatting.inline_repr


__all__ = [
    "testing",
    "unit_registry",
    "setup_registry",
    "PintIndex",
]
