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

import xarray as xr

try:
    from xarray import call_on_dataset
except ImportError:

    def call_on_dataset(func, obj, name, *args, **kwargs):
        if isinstance(obj, xr.DataArray):
            ds = obj.to_dataset(name=name)
        else:
            ds = obj

        result = func(ds, *args, **kwargs)

        if isinstance(obj, xr.DataArray) and isinstance(result, xr.Dataset):
            result = result.get(name).rename(obj.name)

        return result
