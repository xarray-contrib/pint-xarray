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

from . import conversion, formatting


def assert_units_equal(a, b):
    """assert that the units of two xarray objects are equal

    Raises an :py:exc:`AssertionError` if the units of both objects are not
    equal. ``units`` attributes and attached unit objects are compared
    separately.

    Parameters
    ----------
    a, b : DataArray or Dataset
        The objects to compare
    """

    __tracebackhide__ = True

    units_a = conversion.extract_units(a)
    units_b = conversion.extract_units(b)
    assert units_a == units_b, formatting._diff_mapping_repr(
        units_a, units_b, "Units", formatting.summarize_attr
    )

    unit_attrs_a = conversion.extract_unit_attributes(a)
    unit_attrs_b = conversion.extract_unit_attributes(b)
    assert unit_attrs_a == unit_attrs_b, formatting._diff_mapping_repr(
        unit_attrs_a, unit_attrs_b, "Unit attrs", formatting.summarize_attr
    )
