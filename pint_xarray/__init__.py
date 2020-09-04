try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

import pint

from . import testing  # noqa: F401
from . import formatting
from .accessors import PintDataArrayAccessor, PintDatasetAccessor  # noqa: F401
from .accessors import default_registry as unit_registry  # noqa: F401

try:
    __version__ = version("pint-xarray")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"


pint.Quantity._repr_inline_ = formatting.inline_repr
