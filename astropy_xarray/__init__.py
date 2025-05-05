from importlib.metadata import version

import astropy.units

from . import accessors, formatting, testing  # noqa: F401
from .index import AstropyIndex

try:
    __version__ = version("astropy-xarray")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"


astropy.units.Quantity._repr_inline_ = formatting.inline_repr


__all__ = [
    "testing",
    "AstropyIndex",
]
