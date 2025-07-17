# Copyright 2014-2024, xarray developers
# Copyright 2025, Callan Gray

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib.metadata import version

import astropy.units

from astropy_xarray import accessors, formatting, testing  # noqa: F401
from astropy_xarray.index import AstropyIndex

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
