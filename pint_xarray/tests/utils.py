import re
from contextlib import contextmanager

import numpy as np
import pytest
import xarray as xr
from pint.quantity import Quantity
from xarray.testing import assert_equal  # noqa: F401


@contextmanager
def raises_regex(error, pattern):
    __tracebackhide__ = True
    with pytest.raises(error) as excinfo:
        yield
    message = str(excinfo.value)
    if not re.search(pattern, message):
        raise AssertionError(
            f"exception {excinfo.value!r} did not match pattern {pattern!r}"
        )


def array_extract_units(obj):
    if isinstance(obj, (xr.Variable, xr.DataArray, xr.Dataset)):
        obj = obj.data

    try:
        return obj.units
    except AttributeError:
        return None


def extract_units(obj):
    if isinstance(obj, xr.Dataset):
        vars_units = {
            name: array_extract_units(value) for name, value in obj.data_vars.items()
        }
        coords_units = {
            name: array_extract_units(value) for name, value in obj.coords.items()
        }

        units = {**vars_units, **coords_units}
    elif isinstance(obj, xr.DataArray):
        vars_units = {obj.name: array_extract_units(obj)}
        coords_units = {
            name: array_extract_units(value) for name, value in obj.coords.items()
        }

        units = {**vars_units, **coords_units}
    elif isinstance(obj, xr.Variable):
        vars_units = {None: array_extract_units(obj.data)}

        units = {**vars_units}
    elif isinstance(obj, Quantity):
        vars_units = {None: array_extract_units(obj)}

        units = {**vars_units}
    else:
        units = {}

    return units


def attach_units(obj, units):
    if isinstance(obj, xr.DataArray):
        ds = obj._to_temp_dataset()
        new_name = list(ds.data_vars.keys())[0]
        units[new_name] = units.get(obj.name)
        new_ds = attach_units(ds, units)
        new_obj = obj._from_temp_dataset(new_ds)
    elif isinstance(obj, xr.Dataset):
        data_vars = {
            name: attach_units(array.variable, {None: units.get(name)})
            for name, array in obj.data_vars.items()
        }

        coords = {
            name: attach_units(array.variable, {None: units.get(name)})
            for name, array in obj.coords.items()
        }

        new_obj = xr.Dataset(data_vars=data_vars, coords=coords, attrs=obj.attrs)
    elif isinstance(obj, xr.Variable):
        new_data = attach_units(obj.data, units)
        new_obj = obj.copy(data=new_data)
    elif isinstance(obj, Quantity):
        raise ValueError(
            f"cannot attach {units.get(None)} to {obj}: already a quantity"
        )
    else:
        new_obj = Quantity(obj, units.get(None))

    return new_obj


def assert_array_units_equal(a, b):
    __tracebackhide__ = True

    units_a = getattr(a, "units", None)
    units_b = getattr(b, "units", None)

    assert units_a == units_b


def assert_array_equal(a, b):
    __tracebackhide__ = True

    a_ = getattr(a, "magnitude", a)
    b_ = getattr(b, "magnitude", b)

    np.testing.assert_array_equal(a_, b_)


def assert_units_equal(a, b):
    __tracebackhide__ = True
    assert extract_units(a) == extract_units(b)
